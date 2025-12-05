#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CDEARegressor: 双轴注意力回归模型（Conjunction-aware Dual-axis Event Attention）

接口完全对齐当前工程：
- 构造函数: CDEARegressor(args, channel)
    * args: 命令行参数
    * channel: 每个时间步的特征维度 (feature_dim)
- forward(seq_x): 
    * seq_x: (B, seq_len, channel)
    * 输出: (B, 1)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================== CDEA Block ======================

class CDEABlock(nn.Module):
    """
    Conjunction-aware Dual-axis Event Attention Block (CDEA)

    输入:
        x: (B, T, d_model)
           B: batch_size
           T: seq_len (时间步)
        phys_feat: (B, P)，用于构造物理邻域 (例如 miss_distance, n/r/t, t_j2k_inc)
        neighbor_mask: (B, B) bool，可选，True=禁止关注

    逻辑:
        1) 行间注意力：在事件级 embedding 上做 MHA
        2) 列间注意力：在每个样本内部，对时间步序列做 self-attention
    """

    def __init__(
        self,
        d_model: int,
        n_heads_row: int = 4,
        n_heads_col: int = 4,
        dropout: float = 0.1,
        use_neighbor_mask: bool = True,
        k_neighbors: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_neighbor_mask = use_neighbor_mask
        self.k_neighbors = k_neighbors

        # -------- 行间（事件级）注意力 --------
        # 每个样本的 T 个 time tokens -> 一个事件 embedding
        self.event_proj = nn.Linear(d_model, d_model)

        self.row_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads_row,
            dropout=dropout,
            batch_first=True,   # (batch, seq_len, d_model)
        )
        self.row_norm = nn.LayerNorm(d_model)

        # 事件级更新广播回每个 time token 的 MLP
        self.row_broadcast_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.row_gamma = nn.Parameter(torch.tensor(1.0))

        # -------- 列间（时间轴）注意力 --------
        # 在每个样本内部，对 T 个 time tokens 做 self-attention
        self.col_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads_col,
            dropout=dropout,
            batch_first=True,   # (batch=B, seq_len=T, d_model)
        )
        self.col_norm = nn.LayerNorm(d_model)

        # -------- FFN --------
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    @torch.no_grad()
    def build_neighbor_mask_from_phys(self, phys_feat: torch.Tensor) -> torch.Tensor:
        """
        根据物理特征构造邻域 mask:
            - 简单 L2 距离 + top-k KNN
            - 返回 mask: (B, B)，True 表示 "禁止关注"
        """
        B, P = phys_feat.shape
        diff = phys_feat.unsqueeze(1) - phys_feat.unsqueeze(0)   # (B, B, P)
        dist = torch.norm(diff, dim=-1)                          # (B, B)

        k = min(self.k_neighbors, B)
        knn_idx = dist.topk(k=k, dim=-1, largest=False).indices  # (B, k)

        mask = torch.ones(B, B, dtype=torch.bool, device=phys_feat.device)
        arange = torch.arange(B, device=phys_feat.device).unsqueeze(-1)  # (B, 1)
        mask[arange, knn_idx] = False   # 这些邻居允许关注

        mask.fill_diagonal_(False)      # 自己也允许关注
        return mask

    def forward(
        self,
        x: torch.Tensor,                    # (B, T, d_model)
        phys_feat: Optional[torch.Tensor] = None,     # (B, P)
        neighbor_mask: Optional[torch.Tensor] = None  # (B, B) bool
    ) -> torch.Tensor:

        B, T, C = x.shape
        assert C == self.d_model, "CDEABlock: d_model mismatch."

        # ---------- 1) 行间注意力：事件级 ----------
        # 每个样本的时间轴平均池化 -> 事件 embedding
        event_emb = x.mean(dim=1)              # (B, d_model)
        event_emb = self.event_proj(event_emb) # (B, d_model)

        # 邻域 mask: True = 禁止关注
        attn_mask = None
        if self.use_neighbor_mask:
            if neighbor_mask is not None:
                attn_mask = neighbor_mask.to(x.device)
            elif phys_feat is not None:
                attn_mask = self.build_neighbor_mask_from_phys(phys_feat.to(x.device))

        # 把 batch 内所有样本作为一条“事件序列”
        event_seq = event_emb.unsqueeze(0)  # (1, B, d_model)

        if attn_mask is not None:
            event_ctx, _ = self.row_attn(
                event_seq, event_seq, event_seq,
                attn_mask=attn_mask,
                need_weights=False,
            )
        else:
            event_ctx, _ = self.row_attn(
                event_seq, event_seq, event_seq,
                need_weights=False,
            )

        event_ctx = event_ctx.squeeze(0)                    # (B, d_model)
        event_updated = self.row_norm(event_emb + event_ctx)

        # 事件级更新广播到每个时间步
        event_cat = torch.cat([event_emb, event_updated], dim=-1)  # (B, 2*d_model)
        delta = self.row_broadcast_mlp(event_cat)                  # (B, d_model)
        delta = delta.unsqueeze(1).expand(B, T, C)                 # (B, T, d_model)

        x_row = x + self.row_gamma * delta   # (B, T, d_model)

        # ---------- 2) 列间注意力：时间轴 ----------
        col_src = x_row
        col_ctx, _ = self.col_attn(
            col_src, col_src, col_src,
            need_weights=False,
        )
        x_col = self.col_norm(x_row + col_ctx)

        # ---------- 3) FFN ----------
        ffn_out = self.ffn(x_col)
        x_out = self.ffn_norm(x_col + ffn_out)

        return x_out



# ====================== CDEARegressor 主模型 ======================

class CDEARegressor(nn.Module):
    """
    面向当前时序回归任务的 CDEA 模型 (时间步 token 版)

    兼容接口:
        __init__(args, channel):
            - channel = feature_dim = 每个时间步的特征数
        forward(seq_x):
            - seq_x: (B, seq_len, channel)
            - 返回: (B, 1)
    """

    def __init__(self, args, channel: int):
        super().__init__()

        d_model = getattr(args, "d_model", 128)
        num_blocks = getattr(args, "cdea_layers", 2)
        n_heads_row = getattr(args, "cdea_heads_row", 2)
        n_heads_col = getattr(args, "cdea_heads_col", 2)
        dropout = getattr(args, "dropout", 0.1)
        k_neighbors = getattr(args, "cdea_k_neighbors", 8)

        # seq_len 从 args 里取（和 Transformer/LSTM 一致）
        seq_len = getattr(args, "n_latest_cdms", 2) - 1
        if seq_len <= 0:
            raise ValueError("CDEARegressor: n_latest_cdms 至少为 2")

        self.seq_len = seq_len
        self.channel = channel
        self.d_model = d_model

        # 每个时间步: feature_dim -> d_model
        self.time_proj = nn.Linear(channel, d_model)

        # CDEA Blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                CDEABlock(
                    d_model=d_model,
                    n_heads_row=n_heads_row,
                    n_heads_col=n_heads_col,
                    dropout=dropout,
                    # 先关闭物理邻域，稳定后再打开:
                    use_neighbor_mask=False,
                    k_neighbors=k_neighbors,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # 事件级聚合 + 回归头
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        seq_x: torch.Tensor,                    # (B, seq_len, channel)
        phys_feat: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        return_feat: bool = False,              # 是否返回特征向量（用于 SupCR）
    ) -> torch.Tensor:
        """
        Args:
            seq_x: 输入序列 (B, seq_len, channel)
            phys_feat: 物理特征 (B, P)，可选
            neighbor_mask: 邻域mask (B, B)，可选
            return_feat: 是否返回特征向量（用于 SupCR）
        
        Returns:
            y_pred: 预测值 (B, 1)
            feat: 特征向量 (B, d_model)，仅当 return_feat=True 时返回
        """
        B, T, C = seq_x.shape
        assert T == self.seq_len, f"CDEARegressor: seq_len mismatch, got {T}, expect {self.seq_len}"
        assert C == self.channel, f"CDEARegressor: channel mismatch, got {C}, expect {self.channel}"

        # 1) 每个时间步: (B, T, channel) -> (B, T, d_model)
        h = self.time_proj(seq_x)   # (B, T, d_model)

        # 2) 通过多层 CDEA Block
        for blk in self.blocks:
            h = blk(h, phys_feat=phys_feat, neighbor_mask=neighbor_mask)   # (B, T, d_model)

        # 3) 事件级聚合: 在时间轴 T 上求平均（这是特征向量）
        feat = h.mean(dim=1)       # (B, d_model)

        # 4) 回归头
        y_pred = self.reg_head(feat)   # (B, 1)
        
        # 根据参数返回
        if return_feat:
            return y_pred, feat
        return y_pred
