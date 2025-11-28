import torch
import torch.nn as nn
from torchinfo import summary
class FrameTransformer(nn.Module):
    def __init__(self, input_dim=512, model_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super(FrameTransformer, self).__init__()

        # 可学习位置编码（可选用 sinusoidal）
        self.pos_embed = nn.Parameter(torch.randn(1, 120, model_dim))  # 1000 是最大序列长度，可改大

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出映射层：512 → 1
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        """
        x: [B, T, 512]  (变长序列，padding 后统一 T 长度)
        mask: [B, T] -> True 表示要被 mask 掉的位置（padding部分）
        """

        B, T, _ = x.shape
        pos_embed = self.pos_embed[:, :T, :]  # 截取对应长度的位置编码

        x = x + 0.5*pos_embed  # 添加位置编码

        if mask is not None:
            # nn.TransformerEncoder 使用的是 src_key_padding_mask: True 表示忽略该位置
            x = self.transformer(x, src_key_padding_mask=mask)  # [B, T, 512]
        else:
            x = self.transformer(x)

        out = self.head(x)  # [B, T, 1]
        return out

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=1, model_dim=128, num_heads=8, num_layers=2, max_len=120):
        super().__init__()
        self.project = nn.Linear(input_dim, model_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, model_dim))  # [1, T, D]

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # 将时间维聚合
            nn.Flatten(),              # [B, D]
            nn.Linear(model_dim, 1)    # 输出预测值
        )

    def forward(self, x, mask=None):

        # x: [B, T, 1]
        x = self.project(x)  # [B, T, D]
        T = x.size(1)
        pos = self.pos_emb[:, :T, :]  # [1, T, D]

        x = x + 0.5*pos  # 加上位置编码
        if mask is not None:
            x = self.encoder(x, src_key_padding_mask=mask)
        else:
            x = self.encoder(x)

        x = x.transpose(1, 2)  # [B, D, T] for pooling
        out = self.classifier(x)  # [B, 1]
        return out

class LocalContextAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: [B, T, D]
        residual = x
        x, attn_weights  = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + residual)
        x = self.norm2(x + self.ffn(x))
        return x
        # return x,attn_weights

class CAATR_VarLen(nn.Module):
    def __init__(self, dim=512, short_window=4, long_window=12): # 4,12
        super().__init__()
        self.dim = dim
        self.sw = short_window
        self.lw = long_window

        self.short_attn = LocalContextAttention(dim)
        self.long_attn = LocalContextAttention(dim)

        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )

    def forward(self, z, lengths):
        """
        z: Tensor of shape [B, T_max, D], where each sample may have different valid lengths
        lengths: list or tensor of shape [B], containing actual frame count for each sample
        """
        B, T_max, D = z.shape
        device = z.device
        outputs = []

        all_short_attn = []
        all_long_attn = []
        short_weights_all = []
        long_weights_all = []

        for b in range(B):
            T = lengths[b]
            z_b = z[b, :T, :]  # [T, D]
            global_mean = z_b.mean(dim=0, keepdim=True)  # [1, D]
            a_seq = []
            short_self_attn_list = []
            long_self_attn_list = []

            for t in range(T):
                short_start = max(0, t - self.sw)
                short_end = min(T, t + self.sw + 1)
                long_start = max(0, t - self.lw)
                long_end = min(T, t + self.lw + 1)

                short_ctx = z_b[short_start:short_end].unsqueeze(0)  # [1, Ws, D]
                long_ctx = z_b[long_start:long_end].unsqueeze(0)    # [1, Wl, D]

                # short_out,short_weights = self.short_attn(short_ctx)
                # long_out,long_weights = self.long_attn(long_ctx)
                #
                # short_out = short_out[0, t - short_start]
                # long_out = long_out[0, t - long_start]
                #
                # # 注意力 shape: [1, W, W]，取中间位置（即当前帧）的对角值
                # short_diag = short_weights[0].diagonal().mean().item()  # self-attn scalar
                # long_diag = long_weights[0].diagonal().mean().item()
                # short_self_attn_list.append(short_diag)
                # long_self_attn_list.append(long_diag)
                #
                # short_weights_all.append(short_weights.detach().cpu())
                # long_weights_all.append(long_weights.detach().cpu())

                short_out = self.short_attn(short_ctx)[0, t - short_start]
                long_out = self.long_attn(long_ctx)[0, t - long_start]
                residual = z_b[t] - global_mean[0]

                fusion_input = torch.cat([short_out, long_out, residual], dim=-1)
                refined = self.fusion(fusion_input)
                a_seq.append(refined.unsqueeze(0))

            # 返回注意力值列表而非矩阵
            all_short_attn.append(torch.tensor(short_self_attn_list))
            all_long_attn.append(torch.tensor(long_self_attn_list))

            a_seq = torch.cat(a_seq, dim=0)  # [T, D]
            pad_len = T_max - T
            if pad_len > 0:
                pad_tensor = torch.zeros(pad_len, D, device=device)
                a_seq = torch.cat([a_seq, pad_tensor], dim=0)
            outputs.append(a_seq.unsqueeze(0))  # [1, T_max, D]
        return torch.cat(outputs, dim=0)
        # return torch.cat(outputs, dim=0),all_short_attn, all_long_attn,short_weights_all, long_weights_all

if __name__ == "__main__":
    # x = torch.randn(1,5,512)
    # # model = FrameTransformer()
    # model = TransformerClassifier()
    # model = model.to('cuda:0')
    # x = x.to('cuda:0')
    # summary(model, (1, 30,1))
    # summary(model, (1, 30, 512))
    # y = model(x)
    # y = y.squeeze(-1)
    # label = torch.Tensor([[0,0,1,1,1]])
    # label = label.to('cuda:0')
    # loss = nn.BCEWithLogitsLoss()(y, label)
    # print(loss)

    x = torch.randn(1, 84, 512)
    caatr = CAATR_VarLen(dim=512)
    x = x.to('cuda:0')
    caatr = caatr.to('cuda:0')
    out = caatr(x, [30])  # 输出 [B, T_max, D]，无效帧位置保持为 0
    print(out.shape)