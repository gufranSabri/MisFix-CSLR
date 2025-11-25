import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

    
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ContrastiveModel, self).__init__()

        self.llm = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt", 
            cache_dir="./data/models",
            src_lang="en_XX",
            tgt_lang="zh_CN"
        )
        for param in self.llm.parameters():
            param.requires_grad = False

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        self.v2l_proj = nn.Linear(input_dim, self.llm.config.d_model)
        self.l2v_proj = nn.Linear(self.llm.config.d_model, input_dim)

        self.classifier = nn.Linear(input_dim + output_dim, output_dim)

    def contrastive_loss(self, visual_ft, text):
        output_tokens = self.tokenizer(
            text,
            padding="longest",
            return_tensors="pt",
        ).to(visual_ft.device)
        
        with torch.no_grad():
            text_embeds = self.llm.model.encoder.embed_tokens(output_tokens.input_ids)

        image_embeds = visual_ft.mean(1)
        text_embeds = text_embeds.mean(1)

        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale

        batch_size = len(text)
        mask = torch.ones(batch_size, batch_size, device=visual_ft.device, dtype=torch.bool)
        for i in range(batch_size):
            for j in range(batch_size):
                if text[i] == text[j] and i != j:
                    mask[i, j] = False  # mask out identical text labels

        masked_logits = logits_per_text.masked_fill(~mask, float('-inf'))

        return clip_loss(masked_logits)


    def forward(self, x, logits, text):
        x_orig = x.clone()

        x = self.v2l_proj(x)
        cont_loss = self.contrastive_loss(x, text)
        
        x = torch.cat([self.l2v_proj(x)*0.25 + x_orig, logits], dim=-1)
        residual_logits = self.classifier(x)

        return residual_logits, cont_loss
    

class MultiHeadRefiner(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super(MultiHeadRefiner, self).__init__()
        self.heads = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(num_heads)]
        )

    def forward(self, visual_fts):
        residuals = []
        for head in self.heads:
            residual_logits = head(visual_fts)
            residuals.append(residual_logits)
            
        # average of residuals
        residual = sum(residuals) / len(self.heads)

        return residual


class IterativeModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_iterations=3):
        super(IterativeModel, self).__init__()
        self.list = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(num_iterations)]
        )
        self.weight = 1/num_iterations

    def forward(self, visual_fts, logits):
        residuals = []
        prev_logits = logits
        for layer in self.list:
            curr_input = torch.cat([visual_fts, prev_logits], dim=-1)
            residual_logits = layer(curr_input)
            prev_logits = residual_logits + logits
            residuals.append(residual_logits)
            
        # weighted average of residuals
        residuals = [residual * self.weight for residual in residuals]
        residual_logits = sum(residuals)

        return residual_logits