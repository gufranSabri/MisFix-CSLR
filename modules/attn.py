import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)].unsqueeze(1)
    
class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, key_padding_mask=None):
        # x: (T, B, C)
        attn_out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        x = self.norm(q + attn_out)
        return x


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, num_layers=4, slow_stride=8):
        super().__init__()
        self.slow_stride = slow_stride
        self.pos_encoder = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            AttentionLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, q, k, v, key_padding_mask=None):
        q = self.pos_encoder(q)
        k = self.pos_encoder(k)
        v = self.pos_encoder(v)

        # x: (T, B, C)
        for layer in self.layers:
            q = layer(q, k, v, key_padding_mask=key_padding_mask)
        return q