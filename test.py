# test.py
import torch
from transformers import BertTokenizer, BertModel
from data.lrs2_dataset import LRS2Dataset
from models.feature_extractor import MultiScaleFeatureExtractor
from models.adaptive_attention import AdaptiveAttention
from models.temporal_encoder import BiLSTMTemporalEncoder
from models.context_decoder import TransformerDecoderWithContext
from torchvision.transforms import Compose, ConvertImageDtype, Normalize
from jiwer import wer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tokenizer + BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()

# --- Models
extractor = MultiScaleFeatureExtractor().to(device)
attn = AdaptiveAttention(512, 256, 5).to(device)
temporal_encoder = BiLSTMTemporalEncoder(512, 256).to(device)
proj_to_bert = torch.nn.Linear(512, 768).to(device)
decoder = TransformerDecoderWithContext(768, num_heads=8, num_layers=2, vocab_size=tokenizer.vocab_size).to(device)

# --- Load checkpoint
ckpt = torch.load("checkpoints/model_epoch5.pt")
attn.load_state_dict(ckpt["attn"])
temporal_encoder.load_state_dict(ckpt["temporal"])
proj_to_bert.load_state_dict(ckpt["proj"])
decoder.load_state_dict(ckpt["decoder"])

# --- Dataset
transform = Compose([
    ConvertImageDtype(torch.float32),
    Normalize(mean=[0.5], std=[0.5]),
])
val_dataset = LRS2Dataset("data/filelists/val.txt", tokenizer, transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

scales = [1.5, 1.25, 1.0, 0.75, 0.5]
refs, hyps = [], []

# --- Inference Loop
for frames, input_ids, attn_mask in val_loader:
    frames = frames[0].to(device)

    multi_scaled = [
        torch.nn.functional.interpolate(frames, scale_factor=s, mode='bilinear', align_corners=False)
        for s in scales
    ]
    with torch.no_grad():
        features_by_scale = extractor(multi_scaled)

    T = features_by_scale[0].shape[0]
    decoder_hidden = torch.randn(T, 256).to(device)
    fused_features, _ = attn(features_by_scale, decoder_hidden)
    fused_features = fused_features.unsqueeze(0)

    temporal_output = temporal_encoder(fused_features)
    temporal_output_proj = proj_to_bert(temporal_output)

    # Autoregressive decoding
    generated_tokens = [tokenizer.cls_token_id]
    for _ in range(50):
        input_ids_gen = torch.tensor([generated_tokens]).to(device)
        attn_mask_gen = torch.ones_like(input_ids_gen)

        with torch.no_grad():
            bert_embeddings = bert(input_ids=input_ids_gen, attention_mask=attn_mask_gen).last_hidden_state
            logits = decoder(bert_embeddings, memory=temporal_output_proj)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()

        generated_tokens.append(next_token)
        if next_token == tokenizer.sep_token_id:
            break

    pred = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    ref = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    print(f"[GT]   {ref}")
    print(f"[PRED] {pred}")
    refs.append(ref)
    hyps.append(pred)

# --- Evaluate
print(f"\n[RESULT] WER: {wer(refs, hyps):.4f}")
