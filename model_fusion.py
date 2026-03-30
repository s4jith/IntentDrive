import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TrajectoryTransformerFusion(nn.Module):
    def __init__(self, fusion_dim=3):
        super().__init__()
        self.d_model = 64

        # Base kinematic embedding from original model features.
        self.base_embed = nn.Linear(7, self.d_model)

        # Fusion branch: LiDAR/Radar strength features per timestep.
        self.fusion_embed = nn.Linear(fusion_dim, self.d_model)
        self.fusion_ln = nn.LayerNorm(self.d_model)

        self.pos_enc = PositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.social_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True)

        self.K = 3
        self.hidden_dim = 128
        self.future_len = 12

        self.goal_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.K * 2),
        )

        self.traj_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.future_len * 2),
        )

        self.prob_head = nn.Linear(self.hidden_dim, self.K)

    def load_from_base_checkpoint(self, ckpt_path, map_location='cpu'):
        state = torch.load(ckpt_path, map_location=map_location)

        remap = {}
        for k, v in state.items():
            if k.startswith('embed.'):
                remap['base_embed.' + k[len('embed.'):]] = v
            else:
                remap[k] = v

        missing, unexpected = self.load_state_dict(remap, strict=False)
        return missing, unexpected

    def social_pool(self, h_target, neighbor_h_list, device):
        if len(neighbor_h_list) == 0:
            return torch.zeros(self.d_model, device=device), None

        query = h_target.unsqueeze(0).unsqueeze(0)
        neighbor_h_tensor = torch.stack(neighbor_h_list).unsqueeze(0)
        attn_output, attn_weights = self.social_attn(query, neighbor_h_tensor, neighbor_h_tensor)
        return attn_output.squeeze(0).squeeze(0), attn_weights.squeeze(0)

    def forward(self, x, neighbors, fusion_feats=None):
        """
        x: (B, 4, 7)
        neighbors: list length B, each element is list of neighbors with shape (4, 7)
        fusion_feats: (B, 4, F) where F=3 [lidar_pts_norm, radar_pts_norm, sensor_strength]
        """
        B = x.size(0)
        device = x.device

        x_emb = self.base_embed(x)
        if fusion_feats is not None:
            x_emb = self.fusion_ln(x_emb + self.fusion_embed(fusion_feats))

        x_emb = self.pos_enc(x_emb)
        enc_out = self.transformer_encoder(x_emb)
        h = enc_out[:, -1, :]

        final_h = []
        batch_attn_weights = []

        for i in range(B):
            h_target = h[i]
            neighbor_h_list = []

            for n in neighbors[i]:
                n_tensor = torch.as_tensor(n, dtype=torch.float32, device=device).unsqueeze(0)
                n_emb = self.pos_enc(self.base_embed(n_tensor))
                n_enc_out = self.transformer_encoder(n_emb)
                neighbor_h_list.append(n_enc_out[0, -1, :])

            h_social, attn_weights = self.social_pool(h_target, neighbor_h_list, device)
            batch_attn_weights.append(attn_weights)

            h_combined = torch.cat([h_target, h_social], dim=0)
            final_h.append(h_combined)

        h_final = torch.stack(final_h)

        goals = self.goal_head(h_final).view(B, self.K, 2)

        trajs = []
        for k in range(self.K):
            goal_k = goals[:, k, :]
            conditioned_context = torch.cat([h_final, goal_k], dim=1)
            traj_k = self.traj_head(conditioned_context).view(B, 1, self.future_len, 2)
            trajs.append(traj_k)

        traj = torch.cat(trajs, dim=1)

        probs = self.prob_head(h_final)
        probs = torch.softmax(probs, dim=1)

        return traj, goals, probs, batch_attn_weights
