from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization

#############################################################
##   GLAD: Global-Local Attention Decomposition encoder    ##
##   Paper: "Global-Local Attention Decomposition for      ##
##   Terrain Encoding in Humanoid Perceptive Locomotion"   ##
#############################################################


class ActorCriticEncoder(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # Terrain encoder parameters
        map_scan_dim=(33, 21, 3),       # L=33, W=21, 3D xyz coordinates
        mha_dim=64,                      # Embedding dimension D (paper §IV-A)
        num_heads=16,                    # MHA heads (paper §IV-A)
        cnn_downsample=True,             # stride-2 CNN (paper §III-D-1)
        use_global_context=True,         # Enable global attention branch (paper §III-D-2)
        topk: int | None = 32,           # Top-K sparsification K (paper §III-D-2, K=32)
        # Training-side extensions (not in paper, neutral at inference)
        topk_soft_tau: float = 1.0,      # Gumbel-STE temperature during training
        role_disentangle: bool = False,  # Spatial prior bias (off by default)
        global_far_bias: float = 0.6,
        topk_near_bias: float = 1.0,
        topk_gc_exclusion: float = 0.3,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticEncoder.__init__ got unexpected kwargs (ignored): "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.map_scan_dim = map_scan_dim
        self.mha_dim = mha_dim
        self.num_heads = num_heads
        self.L, self.W, self.coord_dim = map_scan_dim
        self.cnn_downsample = cnn_downsample
        self.use_global_context = use_global_context
        self.topk = topk
        self.topk_soft_tau = topk_soft_tau
        self.role_disentangle = role_disentangle
        self.global_far_bias = global_far_bias
        self.topk_near_bias = topk_near_bias
        self.topk_gc_exclusion = topk_gc_exclusion
        self.cnn_output_dim = mha_dim   # CNN output dim == D

        # ── Observation dimensions ───────────────────────────────────────────
        self.obs_groups = obs_groups
        num_actor_obs  = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])

        map_scan_size = self.L * self.W * self.coord_dim
        actor_proprio_dim  = num_actor_obs  - map_scan_size
        critic_proprio_dim = num_critic_obs - map_scan_size
        if actor_proprio_dim <= 0 or critic_proprio_dim <= 0:
            raise ValueError(
                f"proprio_dim error: actor={actor_proprio_dim}, critic={critic_proprio_dim}"
            )
        self.actor_proprio_dim  = actor_proprio_dim
        self.critic_proprio_dim = critic_proprio_dim

        # ── Build modules ────────────────────────────────────────────────────
        self._build_terrain_encoder()

        # Actor / Critic MLP input: [f(D) + c(D) + proprio] or [f(D) + proprio]
        actor_input_dim  = mha_dim + actor_proprio_dim
        critic_input_dim = mha_dim + critic_proprio_dim
        if use_global_context:
            actor_input_dim  += mha_dim   # prepend terrain context c
            critic_input_dim += mha_dim

        self.actor  = MLP(actor_input_dim,  num_actions, actor_hidden_dims,  activation)
        self.critic = MLP(critic_input_dim, 1,           critic_hidden_dims, activation)

        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(actor_proprio_dim) if actor_obs_normalization
            else nn.Identity()
        )
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(critic_proprio_dim) if critic_obs_normalization
            else nn.Identity()
        )

        # ── Action noise ─────────────────────────────────────────────────────
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

        self._print_arch(actor_input_dim, critic_input_dim)

    # ── Module construction ───────────────────────────────────────────────────
    def _build_terrain_encoder(self):
        D = self.mha_dim

        # CNN: paper §III-D-1
        # Layer-1: kernel=5, stride=2, 16 channels  → (L/2)×(W/2) grid
        # Layer-2: kernel=3, stride=1, D channels   → same spatial size
        if self.cnn_downsample:
            self.map_cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, D, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(D),
            )
        else:
            self.map_cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, D, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(D),
            )

        # Proprioceptive embeddings: Linear(proprio → D)
        self.actor_proprio_embedding  = nn.Linear(self.actor_proprio_dim,  D)
        self.critic_proprio_embedding = nn.Linear(self.critic_proprio_dim, D)

        if self.use_global_context:
            # Global branch: u_i = v^T k_i + b_u  (paper eq. 8)
            # Linear(D → 1) produces a scalar logit per local feature
            self.global_pool_selector = nn.Linear(D, 1)

            # Query projector: q = Linear(2D→D)(concat(c, proprio_emb))  (paper §III-D-2)
            self.query_projector = nn.Linear(D * 2, D)
        else:
            self.global_pool_selector = None
            self.query_projector = None

        # Local branch Top-K scorer: s_i = w^T [q; k_i] + b_s  (paper eq. 11)
        if self.topk is not None:
            self.topk_scorer = nn.Linear(D * 2, 1)
        else:
            self.topk_scorer = None

        # MHA: single-query cross-attention over top-K local features
        self.mha = nn.MultiheadAttention(embed_dim=D, num_heads=self.num_heads, batch_first=True)

    def _print_arch(self, actor_in, critic_in):
        N = (self.L // 2 + 1) * (self.W // 2 + 1) if self.cnn_downsample else self.L * self.W
        K = self.topk if self.topk is not None else N
        print(
            f"[GLAD] CNN→{N} tokens | D={self.mha_dim} | heads={self.num_heads} | "
            f"global={'ON' if self.use_global_context else 'OFF'} | "
            f"topK={K if self.topk else 'OFF'}"
        )
        print(f"Actor  MLP in={actor_in}:  {self.actor}")
        print(f"Critic MLP in={critic_in}: {self.critic}")

    # ── Spatial prior (role_disentangle only) ────────────────────────────────
    def _compute_spatial_bias(self, map_scan, feat_h, feat_w):
        """Near/far spatial prior from XY coordinates of the elevation map."""
        xy   = map_scan[..., :2]
        dist = torch.norm(xy, dim=-1, keepdim=True).permute(0, 3, 1, 2)
        dist = F.interpolate(dist, size=(feat_h, feat_w), mode="bilinear", align_corners=False)
        dist_flat = dist.squeeze(1).reshape(dist.shape[0], -1)
        mean = dist_flat.mean(dim=1, keepdim=True)
        std  = dist_flat.std(dim=1,  keepdim=True).clamp_min(1e-6)
        dist_norm = (dist_flat - mean) / std
        return -dist_norm, dist_norm   # near_bias, far_bias

    # ── Core GLAD encoder ─────────────────────────────────────────────────────
    def _encode_terrain(self, obs):
        """
        GLAD forward: CNN → Global-Attention → Top-K Sparsification → MHA.

        Returns
        -------
        encoded_obs     : Tensor[B, actor/critic_input_dim]
        attn_weights    : Tensor[B, 1, K]   — MHA weights over top-K tokens
        topk_indices    : Tensor[B, K] | None
        gc_weights      : Tensor[B, N, 1] | None  — global attention weights
        """
        # ── 1. Reshape map scan ──────────────────────────────────────────────
        # Map stored column-major (W-first); swap W/L in reshape for alignment
        map_scan   = obs[:, -self.L * self.W * self.coord_dim:] \
                         .reshape(-1, self.W, self.L, self.coord_dim)
        height_map = map_scan.permute(0, 3, 1, 2)                # [B, 3, W, L]

        # ── 2. CNN: spatially-aligned local feature extraction ───────────────
        cnn_features         = self.map_cnn(height_map)          # [B, D, H', W']
        feat_h, feat_w       = cnn_features.shape[2], cnn_features.shape[3]
        N_tokens             = feat_h * feat_w                   # 187 with stride-2
        local_features       = (cnn_features.permute(0, 2, 3, 1)
                                             .reshape(-1, N_tokens, self.mha_dim))  # [B, N, D]

        # Spatial prior (only when role_disentangle=True)
        near_bias = far_bias = None
        if self.role_disentangle:
            near_bias, far_bias = self._compute_spatial_bias(map_scan, feat_h, feat_w)

        # ── 3. Proprioceptive embedding ──────────────────────────────────────
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

        # ── 4. Global Attention Branch (paper eq. 8-10) ──────────────────────
        # u_i = v^T k_i + b_u → α = softmax(u) → c = Σ αᵢkᵢ
        gc_weights     = None
        global_context = None
        if self.use_global_context:
            gc_logits = self.global_pool_selector(local_features).squeeze(-1)  # [B, N]
            if self.role_disentangle and far_bias is not None:
                gc_logits = gc_logits + self.global_far_bias * far_bias
            gc_weights     = F.softmax(gc_logits, dim=1).unsqueeze(-1)          # [B, N, 1]
            global_context = torch.bmm(gc_weights.transpose(1, 2),
                                       local_features).squeeze(1)               # [B, D]
            # NOTE: paper uses c directly — no additional MLP transform here

            # Query: q = Linear(2D→D)(concat(c, proprio_emb))  (paper §III-D-2)
            query_vec = self.query_projector(
                torch.cat([global_context, proprio_emb], dim=-1)
            )                                                                    # [B, D]
        else:
            query_vec = proprio_emb                                              # [B, D]

        # ── 5. Top-K Sparsification (paper eq. 11) ───────────────────────────
        # s_i = w^T [q; k_i] + b_s → retain top-K features
        topk_indices = None
        if self.topk is not None:
            N = local_features.shape[1]
            K = min(self.topk, N)

            query_expand = query_vec.unsqueeze(1).expand(-1, N, -1)            # [B, N, D]
            scorer_input = torch.cat([local_features, query_expand], dim=-1)  # [B, N, 2D]
            topk_logits  = self.topk_scorer(scorer_input).squeeze(-1)         # [B, N]

            if self.role_disentangle and near_bias is not None:
                topk_logits = topk_logits + self.topk_near_bias * near_bias
                if gc_weights is not None and self.topk_gc_exclusion > 0:
                    topk_logits = topk_logits \
                                  - self.topk_gc_exclusion * gc_weights.squeeze(-1).detach()

            # Gumbel noise during training for gradient flow through top-K
            if self.training:
                gumbel    = -torch.log(-torch.log(torch.rand_like(topk_logits) + 1e-10) + 1e-10)
                perturbed = topk_logits + gumbel
            else:
                perturbed = topk_logits

            topk_indices = torch.topk(perturbed, K, dim=-1).indices             # [B, K]
            idx_exp      = topk_indices.unsqueeze(-1).expand(-1, -1, self.mha_dim)
            local_sparse = torch.gather(local_features, 1, idx_exp)             # [B, K, D]

            # Straight-through estimator: forward=hard, backward=soft
            if self.training:
                soft_w    = F.softmax(perturbed / max(self.topk_soft_tau, 1e-6), dim=-1)
                soft_topk = torch.gather(soft_w, 1, topk_indices)               # [B, K]
                soft_feat = local_sparse * soft_topk.unsqueeze(-1)
                local_sparse = local_sparse.detach() + (soft_feat - soft_feat.detach())
        else:
            local_sparse = local_features                                        # [B, N, D]

        # ── 6. MHA: local attention over top-K features ──────────────────────
        # query=q, key=value=top-K  →  foothold feature f  (paper §III-D-2)
        mha_out, attn_weights = self.mha(
            query=query_vec.unsqueeze(1),   # [B, 1, D]
            key=local_sparse,               # [B, K, D]
            value=local_sparse,             # [B, K, D]
        )
        foothold_feature = mha_out.squeeze(1)                                   # [B, D]

        # ── 7. Output: concat(f, c, proprio)  (paper §III-D-2 final para) ───
        if self.use_global_context:
            encoded_obs = torch.cat([foothold_feature, global_context, proprio_obs], dim=-1)
        else:
            encoded_obs = torch.cat([foothold_feature, proprio_obs], dim=-1)

        # NaN check disabled for TRT compilation compatibility
        # if torch.isnan(encoded_obs).any() or torch.isinf(encoded_obs).any():
        #     print("[GLAD] Warning: encoded_obs has NaN/Inf")

        return encoded_obs, attn_weights, topk_indices, gc_weights

    # ── Standard ActorCritic interface ───────────────────────────────────────
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self): return self.distribution.mean
    @property
    def action_std(self):  return self.distribution.stddev
    @property
    def entropy(self):     return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        encoded_obs, _, _, _ = self._encode_terrain(obs)
        mean = self.actor(encoded_obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        """
        Actor-only forward for deployment / evaluation.
        Returns (action_mean, attn_weights, topk_indices, gc_weights).
        """
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        encoded_obs, attn_weights, topk_indices, gc_weights = \
            self._encode_terrain(actor_obs)
        return self.actor(encoded_obs), attn_weights, topk_indices, gc_weights

    def evaluate(self, obs, **kwargs):
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        encoded_obs, _, _, _ = self._encode_terrain(critic_obs)
        value = self.critic(encoded_obs)
        if torch.isnan(value).any() or torch.isinf(value).any():
            print("[GLAD] Warning: critic value has NaN/Inf")
        return value

    def get_actor_obs(self, obs):
        return torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs):
        return torch.cat([obs[g] for g in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
