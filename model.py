import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TrajectoryTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 64
        
        # 1. Feature Embedding & Positional Encoding
        self.embed = nn.Linear(7, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model)

        # 2. Transformer Sequence Encoder (Replaces LSTM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Social Attention (Target queries Neighbors)
        self.social_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True)

        self.K = 3  # number of future modes

        # 4. GOAL-CONDITIONED ARCHITECTURE 
        # Base hidden context: Target (64) + Social (64) = 128
        self.hidden_dim = 128
        self.future_len = 12  # Now predicting 6 seconds into future
        
        # Step A: Predict exactly K distinct endpoints (goals)
        self.goal_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.K * 2) # X, Y for K goals
        )

        # Step B: Given the encoded context PLUS a specific Goal, draw the path to get there
        self.traj_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.future_len * 2) # 12 steps to reach the destination
        )

        # 5. Probabilities of each mode
        self.prob_head = nn.Linear(self.hidden_dim, self.K)

    # ----------------------------
    # SOCIAL POOLING
    # ----------------------------
    def social_pool(self, h_target, neighbor_h_list, device):
        if len(neighbor_h_list) == 0:
            return torch.zeros(self.d_model, device=device), None

        # h_target: (64) -> query: (1, 1, 64)
        query = h_target.unsqueeze(0).unsqueeze(0)
        
        # neighbor_h_list: N x 64 -> key, value: (1, N, 64)
        neighbor_h_tensor = torch.stack(neighbor_h_list).unsqueeze(0)
        
        # apply attention
        attn_output, attn_weights = self.social_attn(query, neighbor_h_tensor, neighbor_h_tensor)
        
        return attn_output.squeeze(0).squeeze(0), attn_weights.squeeze(0)

    # ----------------------------
    # FORWARD PASS
    # ----------------------------
    def forward(self, x, neighbors):
        """
        x: (B, 4, 7)
        neighbors: list of length B
        """

        B = x.size(0)
        device = x.device

        # Encode main trajectory sequence with Transformer
        x_emb = self.embed(x)
        x_emb = self.pos_enc(x_emb)
        enc_out = self.transformer_encoder(x_emb)
        h = enc_out[:, -1, :]  # Grab context from last timestep (B, 64)

        final_h = []
        batch_attn_weights = []

        # Loop through batch to handle variable size neighbors
        for i in range(B):
            h_target = h[i]  # (64)

            neighbor_h_list = []
            for n in neighbors[i]:
                n_tensor = torch.tensor(n, dtype=torch.float32, device=device).unsqueeze(0)

                n_emb = self.pos_enc(self.embed(n_tensor))
                n_enc_out = self.transformer_encoder(n_emb)
                
                neighbor_h_list.append(n_enc_out[0, -1, :])  # (64)
                
            # Social attention pooling
            h_social, attn_weights = self.social_pool(h_target, neighbor_h_list, device)
            batch_attn_weights.append(attn_weights)

            # Combine Target and Social context
            h_combined = torch.cat([h_target, h_social], dim=0)  # (128)
            final_h.append(h_combined)

        h_final = torch.stack(final_h)  # (B, 128)

        # ✨ GOAL-CONDITIONED LOGIC ✨
        # 1. Predict Goals (End-points at t=6)
        goals = self.goal_head(h_final)
        goals = goals.view(B, self.K, 2)  # (B, K, 2)
        
        # 2. Condition trajectories on the predicted goals
        trajs = []
        for k in range(self.K):
            goal_k = goals[:, k, :] # Get the k-th destination (B, 2)
            # Concat the base context array with the goal coordinate!
            conditioned_context = torch.cat([h_final, goal_k], dim=1) # (B, 130)
            
            # Predict the path given the condition
            traj_k = self.traj_head(conditioned_context).view(B, 1, self.future_len, 2)
            trajs.append(traj_k)
            
        traj = torch.cat(trajs, dim=1) # (B, K, 12, 2)

        # 3. Mode Probabilities
        probs = self.prob_head(h_final)
        probs = torch.softmax(probs, dim=1)

        return traj, goals, probs, batch_attn_weights