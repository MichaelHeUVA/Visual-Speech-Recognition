# train.py
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from data.lrs2_dataset import LRS2Dataset
from models.feature_extractor import MultiScaleFeatureExtractor
from models.adaptive_attention import AdaptiveAttention
from models.temporal_encoder import BiLSTMTemporalEncoder
from models.context_decoder import TransformerDecoderWithContext
import torch.nn as nn
from torchvision.transforms import Compose, ConvertImageDtype, Normalize
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Config
hidden_dim = 256
max_len = 100
batch_size = 1
num_epochs = 5
scales = [1.5, 1.25, 1.0, 0.75, 0.5]

# --- Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# --- Dataset
transform = Compose([
    ConvertImageDtype(torch.float32),
    Normalize(mean=[0.5], std=[0.5]),
])
train_dataset = LRS2Dataset("data/filelists/train.txt", tokenizer, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# --- Models
extractor = MultiScaleFeatureExtractor().to(device)
attn = AdaptiveAttention(512, hidden_dim, len(scales)).to(device)
temporal_encoder = BiLSTMTemporalEncoder(512, hidden_dim).to(device)
proj_to_bert = nn.Linear(2 * hidden_dim, 768).to(device)
decoder = TransformerDecoderWithContext(768, 8, 2, vocab_size=tokenizer.vocab_size).to(device)
bert = BertModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()

# --- Optimizer & Loss
optimizer = torch.optim.AdamW(
    list(attn.parameters()) + list(temporal_encoder.parameters()) +
    list(proj_to_bert.parameters()) + list(decoder.parameters()),
    lr=2e-4
)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# --- Training Loop
for epoch in range(num_epochs):
    total_loss = 0
    for frames, input_ids, attn_mask in train_loader:
        frames, input_ids, attn_mask = frames[0].to(device), input_ids.to(device), attn_mask.to(device)

        # --- Resize to different scales
        multi_scaled = [
            torch.nn.functional.interpolate(frames, scale_factor=s, mode='bilinear', align_corners=False)
            for s in scales
        ]
        with torch.no_grad():
            features_by_scale = extractor(multi_scaled)

        T = features_by_scale[0].shape[0]
        decoder_hidden = torch.randn(T, hidden_dim).to(device)

        fused_features, _ = attn(features_by_scale, decoder_hidden)
        fused_features = fused_features.unsqueeze(0)

        # Temporal encoding + projection to BERT dim
        temporal_output = temporal_encoder(fused_features)
        temporal_output = proj_to_bert(temporal_output)

        # BERT embeddings of ground truth text
        with torch.no_grad():
            bert_embeds = bert(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state

        # Decode and compute loss
        logits = decoder(bert_embeds, memory=temporal_output)
        loss = criterion(logits.view(-1, logits.shape[-1]), input_ids.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"=== [Epoch {epoch+1}] Avg Loss: {avg_loss:.4f} ===")

    # Save checkpoint
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        "attn": attn.state_dict(),
        "temporal": temporal_encoder.state_dict(),
        "proj": proj_to_bert.state_dict(),
        "decoder": decoder.state_dict()
    }, f"checkpoints/model_epoch{epoch+1}.pt")
