import yaml
import torch
import argparse
# from model import VisualConditionedBERT
from dataset import setup_data

def tokenize_word_level(tokenizer, inputs, labels, vocab, max_length=64, device="cpu"):
    batch_input_ids, batch_label_ids, batch_attention_mask = [], [], []

    blank_id = vocab["<BLANK>"]

    for inp_text, lbl_text in zip(inputs, labels):
        inp_words = inp_text.split()
        lbl_words = lbl_text.split()

        input_ids, label_ids = [], []

        for iw, lw in zip(inp_words, lbl_words):
            inp_tokens = tokenizer(iw, add_special_tokens=False)["input_ids"]
            # label handling
            if lw == "-":
                lbl_tokens = [blank_id] * len(inp_tokens)
            else:
                lbl_tokens = tokenizer(lw, add_special_tokens=False)["input_ids"]
                # pad label or input to match lengths
                if len(lbl_tokens) < len(inp_tokens):
                    lbl_tokens += [blank_id] * (len(inp_tokens) - len(lbl_tokens))
                elif len(lbl_tokens) > len(inp_tokens):
                    inp_tokens += [tokenizer.pad_token_id] * (len(lbl_tokens) - len(inp_tokens))

            input_ids.extend(inp_tokens)
            label_ids.extend(lbl_tokens)


        input_ids = input_ids[:max_length]
        label_ids = label_ids[:max_length]

        attention_mask = [1] * len(input_ids)
        # pad to max_length
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [tokenizer.pad_token_id] * pad_len
            label_ids += [blank_id] * pad_len
            attention_mask += [0] * pad_len

        batch_input_ids.append(input_ids)
        batch_label_ids.append(label_ids)
        batch_attention_mask.append(attention_mask)

    return (
        torch.tensor(batch_input_ids, device=device),
        torch.tensor(batch_label_ids, device=device),
        torch.tensor(batch_attention_mask, device=device),
    )

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

    args.batch_size = 1

    _, loader = setup_data(args)
    model = VisualConditionedBERT(bert_name="google-bert/bert-base-multilingual-cased", video_dim=2304, num_classes=1000)

    for idx, batch in enumerate(loader):
        visual_fts, vid_lgt, cslr_label, cslr_label_lengths, label_text, errored_label_texts, pred_label_texts, video_names = batch

        inp = {
            "input_texts": errored_label_texts,
            "video_feats": visual_fts.permute(0, 2, 1),  # (B, T, D) -> (B, D, T)
            "labels": label_text,
            "vid_lgt": vid_lgt,
            "cslr_labels": cslr_label,
            "cslr_label_lgt": cslr_label_lengths
        }

        tokenized = tokenize_word_level(
            tokenizer=model.tokenizer,
            inputs=inp["input_texts"],
            labels=inp["labels"],
            gloss_dict=model.tokenizer.get_vocab(),
            device="cpu"
        )
        print(inp["input_texts"])
        print(inp["labels"])

        print()

        print(tokenized[0].shape)
        print(tokenized[1].shape)
        print(tokenized[2].shape)

        print()

        # decode and print each token
        for i in range(tokenized[0].shape[0]):
            input_ids = tokenized[0][i].tolist()
            label_ids = tokenized[1][i].tolist()
            attention_mask = tokenized[2][i].tolist()

            input_tokens = model.tokenizer.convert_ids_to_tokens(input_ids)
            label_tokens = model.tokenizer.convert_ids_to_tokens(label_ids)

            print("Input Tokens: ", input_tokens)
            print("Label Tokens: ", label_tokens)
            print("Attention Mask: ", attention_mask)
            print()


        break
        
    