# 在你的主脚本或 CAMEF 模型文件中添加以下代码

import torch
import torch.nn as nn
from module import (  # 从你的 module.py 导入
    FixedEmbedding,
    TransformerBlock,
    RMSNorm,
    BSQuantizer,
    HierarchicalEmbedding,
    DualHead
)

class KronosEncoder(nn.Module):
    def __init__(self, seq_len, d, d_model=256, n_layers=4, n_heads=4, ff_dim=1024, s1_bits=8, s2_bits=8):
        super().__init__()
        self.seq_len = seq_len
        self.d = d
        self.d_model = d_model

        # Step 1: 线性投影输入到 d_model 维
        self.input_proj = nn.Linear(d, d_model)

        # Step 2: 位置编码（使用 FixedEmbedding）
        self.pos_embedding = FixedEmbedding(seq_len, d_model)

        # Step 3: 多层 Transformer 编码器
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, attn_dropout_p=0.1, resid_dropout_p=0.1)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        # Step 4: 可选：向量量化（BSQuantizer）——如果你希望离散化表示
        # self.bsq = BSQuantizer(s1_bits=s1_bits, s2_bits=s2_bits, beta=0.25, gamma0=1.0, gamma=0.1, zeta=0.1, group_size=8)
        # 但如果你只想要连续 embedding，可以跳过量化

        # Step 5: 全局平均池化 + 投影到 1024
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [B, d_model, L] -> [B, d_model, 1]
        self.proj_out = nn.Linear(d_model, 1024)

    def forward(self, x):
        """
        x: [B, seq_len, d]
        returns: [B, 1024]
        """
        B, L, d = x.shape
        assert L == self.seq_len and d == self.d

        # 投影到 d_model
        x = self.input_proj(x)  # [B, L, d_model]

        # 加上位置编码
        pos = self.pos_embedding(torch.arange(L, device=x.device)).unsqueeze(0)  # [1, L, d_model]
        x = x + pos

        # Transformer 编码
        for layer in self.layers:
            x = layer(x)  # [B, L, d_model]
        x = self.norm(x)

        # 转换为 [B, d_model, L] 以便池化
        x = x.transpose(1, 2)  # [B, d_model, L]
        x = self.global_pool(x).squeeze(-1)  # [B, d_model]

        # 投影到 1024
        x = self.proj_out(x)  # [B, 1024]
        return x
