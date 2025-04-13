import torch
import torch.nn as nn

class BiLSTMTemporalEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, dropout=0.1):
        super(BiLSTMTemporalEncoder, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        """
        x: [B, T, D]
        returns: [B, T, 2*H]
        """
        output, _ = self.bilstm(x)
        return output