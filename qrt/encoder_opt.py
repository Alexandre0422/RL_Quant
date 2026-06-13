# -*- coding: utf-8 -*-
"""
encoder_opt.py — P3: GLAD encoder 前向的数学等价重写(monkeypatch,不改模型文件)
══════════════════════════════════════════════════════════════════════════════
基于 H 配置 profile(单步 1.53ms)的精确制导,消除当前最大块 GPS/scorer 链
(~0.73ms: cat 物化 0.114 + scorer GEMM(K=128) 0.234 + GPS gemv 0.121 +
softmax/bmm 融合块 0.257)与 MHA 分解慢路径(~0.2ms 碎 kernel):

① 代数重写(对 Linear 的精确恒等式,非近似):
     topk_scorer(cat([local, query_expand]))
   ≡ local @ Wₛ[:, :D]ᵀ + (query @ Wₛ[:, D:]ᵀ + bₛ)        ← cat 物化消失
   且 scorer 的 local 半部与 GPS selector 合并为单个 [D→2] GEMM:
     logits2 = local @ cat([W_gps, Wₛ[:, :D]])ᵀ             ← 98MB 只读一次
   (两列均只依赖 local_features,可在 query 产生之前计算)

② MHA 显式重写: nn.MultiheadAttention(need_weights=True 走 python 分解
   慢路径)→ 手写 single-query cross-attention(H=16, head_dim=4, K=32),
   权重直接取 in_proj_weight/out_proj(同一参数对象,训练更新自动生效),
   紧凑张量形式交给 inductor 融合。attn_weights 语义与原版一致
   (average_attn_weights=True → [B,1,K])。

用法:
    from qrt.encoder_opt import patch_encoder, unpatch_encoder
    patch_encoder(model)        # 在 InferenceRunner 编译之前调用
    ...
    unpatch_encoder(model)      # 还原(消融对照)

约束: 仅支持 use_global_context=True 且 topk 非 None 的 GLAD 主配置
(其它配置 patch_encoder 直接拒绝);role_disentangle 必须为 False;
training 分支(Gumbel+STE)完整保留,数学与原实现逐项一致。
"""
from __future__ import annotations

import types
import warnings

import torch
import torch.nn.functional as F


def _encode_terrain_opt(self, obs):
    """数学等价优化版(对照 rsl_rl/modules/actor_critic_encoder.py:_encode_terrain)。"""
    # ── 1. map reshape(与原版一致)─────────────────────────────────────────
    map_scan = obs[:, -self.L * self.W * self.coord_dim:] \
                  .reshape(-1, self.W, self.L, self.coord_dim)
    height_map = map_scan.permute(0, 3, 1, 2)

    # ── 2. CNN ───────────────────────────────────────────────────────────────
    cnn_features = self.map_cnn(height_map)
    feat_h, feat_w = cnn_features.shape[2], cnn_features.shape[3]
    N_tokens = feat_h * feat_w
    local_features = (cnn_features.permute(0, 2, 3, 1)
                                  .reshape(-1, N_tokens, self.mha_dim))   # [B,N,D]
    B, N, D = local_features.shape

    # ── 3. proprio embedding(与原版一致)───────────────────────────────────
    proprio_obs = obs[:, :-self.L * self.W * self.coord_dim]
    if proprio_obs.shape[1] == self.actor_proprio_dim:
        proprio_emb = self.actor_proprio_embedding(proprio_obs)
    elif proprio_obs.shape[1] == self.critic_proprio_dim:
        proprio_emb = self.critic_proprio_embedding(proprio_obs)
    else:
        raise ValueError(
            f"proprio dim {proprio_obs.shape[1]} matches neither "
            f"actor({self.actor_proprio_dim}) nor critic({self.critic_proprio_dim})"
        )

    # ── 4+5a. ① GPS selector 与 scorer-local 合并为单 GEMM([D→2])──────────
    w2 = torch.cat([self.global_pool_selector.weight,        # [1,D]
                    self.topk_scorer.weight[:, :D]], dim=0)  # [1,D] ← cat 后前 D 列
    logits2 = local_features @ w2.t()                        # [B,N,2] 98MB 只读一次
    gc_logits = logits2[..., 0] + self.global_pool_selector.bias   # [B,N]
    score_local = logits2[..., 1]                                  # [B,N](暂存)

    gc_weights = F.softmax(gc_logits, dim=1).unsqueeze(-1)         # [B,N,1]
    global_context = torch.bmm(gc_weights.transpose(1, 2),
                               local_features).squeeze(1)          # [B,D]
    query_vec = self.query_projector(
        torch.cat([global_context, proprio_emb], dim=-1))          # [B,D]

    # ── 5b. ① scorer-query 半部(标量广播,cat 物化消除)─────────────────────
    score_query = (query_vec @ self.topk_scorer.weight[:, D:].t()
                   + self.topk_scorer.bias)                        # [B,1]
    topk_logits = score_local + score_query                        # [B,N] 广播

    K = min(self.topk, N)
    if self.training:                                              # Gumbel(与原版一致)
        gumbel = -torch.log(-torch.log(torch.rand_like(topk_logits) + 1e-10) + 1e-10)
        perturbed = topk_logits + gumbel
    else:
        perturbed = topk_logits

    topk_indices = torch.topk(perturbed, K, dim=-1).indices        # [B,K]
    idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, self.mha_dim)
    local_sparse = torch.gather(local_features, 1, idx_exp)        # [B,K,D]

    if self.training:                                              # STE(与原版一致)
        soft_w = F.softmax(perturbed / max(self.topk_soft_tau, 1e-6), dim=-1)
        soft_topk = torch.gather(soft_w, 1, topk_indices)
        soft_feat = local_sparse * soft_topk.unsqueeze(-1)
        local_sparse = local_sparse.detach() + (soft_feat - soft_feat.detach())

    # ── 6. ② 手写 single-query cross-attention(≡ nn.MHA, dropout=0)────────
    W_in = self.mha.in_proj_weight                                 # [3D,D]
    b_in = self.mha.in_proj_bias                                   # [3D]
    H = self.num_heads
    hd = D // H
    q = query_vec @ W_in[:D].t() + b_in[:D]                        # [B,D]
    k = local_sparse @ W_in[D:2 * D].t() + b_in[D:2 * D]           # [B,K,D]
    v = local_sparse @ W_in[2 * D:].t() + b_in[2 * D:]             # [B,K,D]
    qh = q.view(B, H, 1, hd)                                       # [B,H,1,hd]
    kh = k.view(B, K, H, hd).transpose(1, 2)                       # [B,H,K,hd]
    vh = v.view(B, K, H, hd).transpose(1, 2)                       # [B,H,K,hd]
    scores = (qh @ kh.transpose(-1, -2)) * (1.0 / hd ** 0.5)       # [B,H,1,K]
    attn = F.softmax(scores, dim=-1)
    oh = attn @ vh                                                 # [B,H,1,hd]
    foothold_feature = (oh.reshape(B, D) @ self.mha.out_proj.weight.t()
                        + self.mha.out_proj.bias)                  # [B,D]
    attn_weights = attn.squeeze(2).mean(dim=1, keepdim=True)       # [B,1,K](头平均,同原版)

    # ── 7. 输出(与原版一致)────────────────────────────────────────────────
    encoded_obs = torch.cat([foothold_feature, global_context, proprio_obs], dim=-1)
    return encoded_obs, attn_weights, topk_indices, gc_weights


def patch_encoder(model):
    """以数学等价优化版替换 model._encode_terrain(实例级,不改类/源文件)。"""
    if not (model.use_global_context and model.topk is not None):
        warnings.warn("[qrt] encoder_opt 仅支持 GLAD 主配置(gc+topk),未 patch")
        return model
    if model.role_disentangle:
        warnings.warn("[qrt] role_disentangle=True 未支持,未 patch")
        return model
    if getattr(model, "_encode_terrain_orig", None) is None:
        model._encode_terrain_orig = model._encode_terrain
    model._encode_terrain = types.MethodType(_encode_terrain_opt, model)
    return model


def unpatch_encoder(model):
    """还原原始 _encode_terrain。"""
    orig = getattr(model, "_encode_terrain_orig", None)
    if orig is not None:
        model._encode_terrain = orig
        model._encode_terrain_orig = None
    return model
