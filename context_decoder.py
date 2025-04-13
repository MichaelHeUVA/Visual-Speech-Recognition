# filename: context_decoder.py
import torch
import torch.nn as nn

class TransformerDecoderWithContext(nn.Module):
    def __init__(self, d_model=768, num_heads=8, num_layers=3, dim_feedforward=2048, dropout=0.1, vocab_size=100):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_bert_embeds, memory, tgt_mask=None, memory_mask=None):
        """
        tgt_bert_embeds: [B, T_dec, D] from BERT
        memory: [B, T_enc, D] from visual encoder
        """
        tgt_emb = tgt_bert_embeds  # already encoded
        output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        return self.fc_out(output)  # [B, T_dec, vocab_size]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
