import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_scales):
        super(AdaptiveAttention, self).__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim

        self.scale_projections = nn.ModuleList([
            nn.Linear(feature_dim, hidden_dim) for _ in range(num_scales)
        ])

        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, features_by_scale, decoder_hidden_state):
        """
        features_by_scale: List of [T, D] tensors from each scale
        decoder_hidden_state: [T, H] tensor from decoder
        """
        T = features_by_scale[0].shape[0]

        # Project each scale's feature to attention space
        projected_scales = []  # Will be list of [T, H]
        for i in range(self.num_scales):
            projected = self.scale_projections[i](features_by_scale[i])  # [T, H]
            projected_scales.append(projected)

        # Stack all scales: [T, num_scales, H]
        stacked_proj = torch.stack(projected_scales, dim=1)

        # Expand decoder hidden state: [T, H] → [T, num_scales, H]
        dec_proj = self.hidden_proj(decoder_hidden_state).unsqueeze(1).expand_as(stacked_proj)

        # Compute attention weights
        attn_input = torch.tanh(stacked_proj + dec_proj)  # [T, num_scales, H]
        attn_scores = self.attn_score(attn_input).squeeze(-1)  # [T, num_scales]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [T, num_scales]

        # Fuse features
        stacked_feats = torch.stack(features_by_scale, dim=1)  # [T, num_scales, D]
        weighted = (attn_weights.unsqueeze(-1) * stacked_feats).sum(dim=1)  # [T, D]

        return weighted, attn_weights  # optionally return weights for inspection
