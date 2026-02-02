import torch
import math
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        # x: (T, B, C)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm(x + attn_out)
        return x


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, num_layers=4):
        super().__init__()

        self.layers = nn.ModuleList([
            AttentionLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        # x: (T, B, C)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x

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
        # x: (T, B, C)
        return x + self.pe[:x.size(0)].unsqueeze(1)

class Local(nn.Module):
    def __init__(self, input_size, hidden_size, ks=5, dropout=0.1):
        super().__init__()

        self.proj = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.temporal = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=ks,
            stride=1,
            padding=ks // 2
        )

        self.norm2 = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lgt):
        x = self.proj(x)
        x = self.norm1(x)

        residual = x
        x = x.transpose(1, 2)          # [B, H, T]
        x = self.temporal(x)
        x = x.transpose(1, 2)          # [B, T, H]

        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x + residual

        if lgt is not None:
            mask = torch.arange(x.size(1), device=x.device)[None, :] < lgt[:, None]
            x = x * mask.unsqueeze(-1)

        return x
        

class PoseEncoder(nn.Module):
    def __init__(self, cfg, hidden_size):
        super(PoseEncoder, self).__init__()
        self.cfg = cfg

        assert hidden_size % 4 == 0, "hidden_size must be divisible by 4"
        
        # LEVEL 1
        self.local_left_l1 = Local(
            input_size=len(self.cfg['left'])*3,
            hidden_size=hidden_size//4,
            ks=5, dropout=0.1
        )
        self.local_right_l1 = Local(
            input_size=len(self.cfg['right'])*3,
            hidden_size=hidden_size//4,
            ks=5, dropout=0.1
        )
        self.local_face_l1 = Local(
            input_size=len(self.cfg['face'])*3,
            hidden_size=hidden_size//4,
            ks=5, dropout=0.1
        )
        self.local_body_l1 = Local(
            input_size=len(self.cfg['body'])*3,
            hidden_size=hidden_size//4,
            ks=5, dropout=0.1
        )

        # LEVEL 2
        self.fusion_lr_l2 = Local(
            input_size=hidden_size//2,
            hidden_size=hidden_size//2,
            ks=5, dropout=0.1
        )
        self.fusion_bf_l2 = Local(
            input_size=hidden_size//2,
            hidden_size=hidden_size//2,
            ks=5, dropout=0.1
        )

        # LEVEL 3
        self.fusion_all_l3 = Local(
            input_size=hidden_size,
            hidden_size=hidden_size,
            ks=5, dropout=0.1
        )

        self.pos_enc = PositionalEncoding(hidden_size)

        self.attn = Attention(
            d_model=hidden_size,
            num_heads=4,
            num_layers=4,
        )

    def create_mask(self, seq_lengths: list[int], device="cpu"):
        lengths = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
        max_len = lengths.max().item()
        range_row = torch.arange(max_len, dtype=torch.int32, device=device).expand(len(lengths), -1)
        lengths = lengths.unsqueeze(1)
        mask = range_row < lengths  # shape: (batch_size, max_len)
        return mask

    def prepare_pose(self, x):
        x = x.permute(0,2,1,3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x

    def forward(self, x, lgt):
        x = x['keypoint']
        left = self.prepare_pose(x[:, :, :, self.cfg['left']])
        right = self.prepare_pose(x[:, :, :, self.cfg['right']])
        face = self.prepare_pose(x[:, :, :, self.cfg['face']])
        body = self.prepare_pose(x[:, :, :, self.cfg['body']])

        left = self.local_left_l1(left, lgt)
        right = self.local_right_l1(right, lgt)
        face = self.local_face_l1(face, lgt)
        body = self.local_body_l1(body, lgt)

        lr = torch.cat([left, right], dim=-1)
        bf = torch.cat([body, face], dim=-1)
        
        lr = self.fusion_lr_l2(lr, lgt)
        bf = self.fusion_bf_l2(bf, lgt)

        x = torch.cat([lr, bf], dim=-1)
        x = self.fusion_all_l3(x, lgt)

        x = self.pos_enc(x)
        mask = self.create_mask(lgt.tolist(), device=x.device)
        x = self.attn(x, key_padding_mask=~mask) + x

        return x