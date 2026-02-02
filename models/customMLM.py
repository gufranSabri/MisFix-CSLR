
import yaml
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.tconv import TemporalConv
from modules.BiLSTM import BiLSTMLayer
from modules.attn import Attention
from modules.seqkd import SeqKD
from modules.wer import beam_decode
from modules.contrastive import clip_loss
from models.funnel import PoseEncoder
from dataset import setup_data

from transformers import BertModel, BertTokenizer

class CSLR_Head(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.tconv = TemporalConv(input_size, hidden_size, num_classes=num_classes)
        self.lstm = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lgt):
        conv_op = self.tconv(x, lgt)
        x, lgt = conv_op['visual_feat'], conv_op['feat_len']
        fts = self.lstm(x, lgt)['predictions']

        logits = self.classifier(fts)

        return logits, fts.permute(1, 0, 2), conv_op
    

class CustomMLM(nn.Module):
    def __init__(self, arg, gloss_dict, num_classes_cslr_head):
        super().__init__()
        self.include_pose = arg.include_pose

        self.bert = BertModel.from_pretrained(arg.llm)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<BLANK>"]})
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.gloss_dict = gloss_dict
        self.gloss_dict_inv = {v:k for k,v in gloss_dict.items()}

        for k, v in self.bert.named_parameters():
            v.requires_grad = False

        self.slr_head = CSLR_Head(arg.feature_input_dim, self.bert.config.hidden_size, num_classes=num_classes_cslr_head)
        
        if self.include_pose:
            self.pose_encoder = PoseEncoder(cfg=arg.pose_encoder_cfg, hidden_size=self.bert.config.hidden_size) if self.include_pose else None
            self.slr_head_pose = CSLR_Head(self.bert.config.hidden_size, self.bert.config.hidden_size, num_classes=num_classes_cslr_head)
            self.pft_cross_attention = Attention(d_model=self.bert.config.hidden_size, num_heads=8, num_layers=2)
            self.ptft_cross_attention = Attention(d_model=self.bert.config.hidden_size, num_heads=8, num_layers=2)

        self.self_attention = Attention(d_model=self.bert.config.hidden_size, num_heads=8, num_layers=2)
        self.vft_cross_attention = Attention(d_model=self.bert.config.hidden_size, num_heads=8, num_layers=2)

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        self.mlm_head = nn.Linear(self.bert.config.hidden_size, num_classes_cslr_head)

        # self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.ctc_loss = torch.nn.CTCLoss(reduction='none', zero_infinity=True, blank=0)
        self.distillation = SeqKD(T=8)

    def create_mask(self, seq_lengths: list[int], device="cpu"):
        lengths = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
        max_len = lengths.max().item()
        range_row = torch.arange(max_len, dtype=torch.int32, device=device).expand(len(lengths), -1)
        lengths = lengths.unsqueeze(1)
        mask = range_row < lengths  # shape: (batch_size, max_len)
        return mask
    
    def compute_ctc_loss(self, ret_dict, label, label_lgt):
        loss = 0
        loss_config = {'SeqCTC': 1.0, 'ConvCTC': 1.0, 'Dist': 25.0}
        for k, weight in loss_config.items():
            if k == 'ConvCTC':
                loss += weight * self.ctc_loss(
                    ret_dict["conv_logits"].cpu().log_softmax(-1),
                    label.cpu().int(), 
                    ret_dict["feat_len"].cpu().int(),
                    label_lgt.cpu().int()
                ).mean() if ret_dict["conv_logits"] is not None else 0.0
                
            elif k == 'SeqCTC':
                loss += weight * self.ctc_loss(
                    ret_dict["sequence_logits"].cpu().log_softmax(-1),
                    label.cpu().int(), 
                    ret_dict["feat_len"].cpu().int(),
                    label_lgt.cpu().int()
                ).mean()

            elif k == 'Dist':
                loss += weight * self.distillation(
                    ret_dict["conv_logits"].cpu(),
                    ret_dict["sequence_logits"].detach().cpu(),
                    use_blank=False
                ) if ret_dict["conv_logits"] is not None else 0.0

        return loss
    
    def compute_ce_loss(self, logits, labels, logit_lgt, label_lgt):
        batch_size = logits.size(0)
        ce_loss = 0.0
        for i in range(batch_size):
            logit_len = logit_lgt[i]
            label_len = label_lgt[i]

            ce_loss += self.ce(
                logits[i, :logit_len, :],
                labels[i, :label_len]
            )
            
        return ce_loss
    
    def gloss_embeddings(self, sentence, device):
        words = sentence.split(" ")
        word_embeddings = []

        with torch.no_grad():
            for word in words:
                encoded = self.tokenizer(
                    word,
                    return_tensors="pt",
                    add_special_tokens=True
                )

                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                hidden_states = outputs.last_hidden_state.squeeze(0)
                subword_embeddings = hidden_states[1:-1] # remove [CLS] and [SEP]
                word_embedding = subword_embeddings.mean(dim=0)

                word_embeddings.append(word_embedding)

        return torch.stack(word_embeddings, dim=0)

    def compute_contrastive_loss(self, visual, text):
        output_tokens = self.tokenizer(
            text,
            padding="longest",
            return_tensors="pt",
        ).to(visual.device)

        with torch.no_grad():
            text_embeds = self.bert(
                input_ids=output_tokens["input_ids"],
                attention_mask=output_tokens["attention_mask"]
            ).last_hidden_state
        
        image_embeds = visual.mean(1)  # global pooling
        text_embeds = text_embeds.mean(1)  # global pooling

        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale

        return clip_loss(logits_per_text)


    def forward(self, inp):
        input_embeddings, lengths = [], []
        for sentence in inp["input_texts"]:
            emb = self.gloss_embeddings(sentence, inp["visual_fts"].device)
            input_embeddings.append(emb)
            lengths.append(emb.size(0))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_embeddings, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        mask = self.create_mask(lengths, device=inp["visual_fts"].device)

        sa_out = self.self_attention(
            q=input_ids,
            k=input_ids,
            v=input_ids,
            key_padding_mask=~mask
        )

        rgb_logits, fts, conv_op = self.slr_head(inp["visual_fts"], inp["vid_lgt"])   # B × H
        mask = self.create_mask(seq_lengths=conv_op['feat_len'].tolist(), device=input_ids.device)

        pose_logits, pose_fts, pose_conv_op = None, None, None
        if self.include_pose:
            pose_features = self.pose_encoder(inp["pose_fts"], inp["pose_lgt"])
            pose_logits, pose_fts, pose_conv_op = self.slr_head_pose(pose_features.permute(0, 2, 1), inp["pose_lgt"])  

            sa_out = self.pft_cross_attention(
                q=sa_out,
                k=pose_features,
                v=pose_features,
                key_padding_mask=~self.create_mask(inp["pose_lgt"].tolist(), device=inp["visual_fts"].device)
            ) + sa_out

            sa_out = self.ptft_cross_attention(
                q=sa_out,
                k=pose_fts,
                v=pose_fts,
                key_padding_mask=~self.create_mask(pose_conv_op["feat_len"].tolist(), device=inp["visual_fts"].device)
            ) + sa_out

        fused = self.vft_cross_attention(
            q=sa_out,
            k=fts,
            v=fts,
            key_padding_mask=~mask
        ) + sa_out
        
        logits = self.mlm_head(fused)

        if self.training:
            loss = self.compute_ctc_loss(
                ret_dict = {
                    "conv_logits": conv_op['conv_logits'],
                    "sequence_logits": rgb_logits,
                    "feat_len": conv_op['feat_len'],
                },
                label = inp["cslr_labels"],
                label_lgt = inp["cslr_label_lgt"]
            )

            loss += self.compute_ctc_loss(
                ret_dict={
                    "conv_logits": pose_conv_op['conv_logits'],
                    "sequence_logits": pose_logits,
                    "feat_len": pose_conv_op['feat_len'],
                },
                label = inp["cslr_labels"],
                label_lgt = inp["cslr_label_lgt"]
            )

            loss += self.compute_ctc_loss(
                ret_dict={
                    "conv_logits": None,
                    "sequence_logits": logits.permute(1, 0, 2),
                    "feat_len": torch.tensor(lengths, dtype=torch.long, device=logits.device)
                },
                label = inp["cslr_labels"],
                label_lgt = inp["cslr_label_lgt"]
            )

            loss += self.compute_contrastive_loss(fused, inp["label_texts"]).to(loss.device)

            # labels = []
            # prev_index = 0
            # for label_lgt in inp["modified_lengths"]:
            #     lgt = label_lgt.item()
            #     labels.append(inp["accounted_labels"][prev_index:prev_index+lgt])
            #     prev_index += lgt
            # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

            # loss += self.compute_ce_loss(logits, labels, lengths, inp["modified_lengths"])
            
            return loss #+ ce_loss
        
        else:
            decoded = beam_decode(logits.permute(1, 0, 2).cpu(), self.gloss_dict, torch.tensor(lengths, dtype=torch.long).cpu())
            rgb_stream_decoded = beam_decode(rgb_logits.cpu(), self.gloss_dict, conv_op['feat_len'].cpu())
            pose_stream_decoded = beam_decode(pose_logits.cpu(), self.gloss_dict, pose_conv_op['feat_len'].cpu())

            return inp["label_texts"], decoded, rgb_stream_decoded, pose_stream_decoded
        

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',dest='config', default='./configs/baseline.yaml')
    parser.add_argument('--dataset',dest='dataset', default='phoenix2014-T')
    args=parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    args.__dict__.update(config)
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        config = yaml.safe_load(f)
    args.__dict__.update(config)

    args.batch_size = 2

    _, loader = setup_data(args)
    model = CustomMLM(bert_name="google-bert/bert-base-multilingual-cased", video_dim=2304, num_classes_rgb_head=loader.dataset.num_classes, gloss_dict=loader.dataset.gloss_dict)
    model.eval()

    for idx, batch in enumerate(loader):
        visual_fts, vid_lgt, \
        cslr_label, cslr_label_lengths, errored_label_text, \
        accounted_label, accounted_label_text, pred_label_text, \
        modified_lengths = batch

        inp = {
            "visual_fts": visual_fts.permute(0, 2, 1),  # (B, T, D) -> (B, D, T)
            "vid_lgt": vid_lgt,
            
            "cslr_labels": cslr_label,
            "cslr_label_lgt": cslr_label_lengths,

            "input_texts": errored_label_text if model.training else pred_label_text,

            "accounted_labels": accounted_label,
            "accounted_label_text": accounted_label_text,
            "modified_lengths": torch.tensor(modified_lengths, dtype=torch.long)
        }

        loss = model(inp)
        print(loss)
    
