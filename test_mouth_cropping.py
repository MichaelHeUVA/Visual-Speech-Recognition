import cv2
import torch
import torchvision
from torchvision import transforms
from data_loader import AVSRDataLoader
from models.feature_extractor import MultiScaleFeatureExtractor
from detector import LandmarksDetector
from models.adaptive_attention import AdaptiveAttention
from models.temporal_encoder import BiLSTMTemporalEncoder
from models.context_decoder import TransformerDecoderWithContext
from transformers import BertTokenizer, BertModel

def save2vid(filename, vid, frames_per_second):
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.io.write_video(filename, vid, int(frames_per_second))


# def preprocess_video(src_filename, dst_filename):
#     landmarks = landmarks_detector(src_filename)
#     data = dataloader.load_data(src_filename, landmarks)
#     fps = cv2.VideoCapture(src_filename).get(cv2.CAP_PROP_FPS)
#     save2vid(dst_filename, data, fps)
#     return

def preprocess_video_multiscale(src_filename, dst_prefix, scales=[1.0, 0.75, 0.5, 0.33]):
    landmarks = landmarks_detector(src_filename)
    fps = cv2.VideoCapture(src_filename).get(cv2.CAP_PROP_FPS)

    for scale in scales:
        # Update scale for the dataloader
        dataloader.set_scale(scale)  # Make sure your loader supports this
        data = dataloader.load_data(src_filename, landmarks)

        scaled_filename = f"{dst_prefix}_scale_{int(scale * 100)}.mp4"
        save2vid(scaled_filename, data, fps)
        print(f"[INFO] Saved scaled video: {scaled_filename}")


dataloader = AVSRDataLoader(
    speed_rate=1,
    convert_gray=False,
)
landmarks_detector = LandmarksDetector()

preprocess_video_multiscale(
    src_filename="videos/clip.mp4",
    dst_prefix="videos/roi_clip",
    scales=[1.5, 1.25, 1.0, 0.75, 0.5]
)

to_tensor = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),  # Convert uint8 → float32
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Optional normalization
])

def load_video_tensor(path):
    frames, _, _ = torchvision.io.read_video(path, pts_unit="sec")
    frames = frames.permute(0, 3, 1, 2)  # [T, H, W, C] → [T, C, H, W]
    return to_tensor(frames)

video_paths = [
    "videos/roi_clip_scale_150.mp4",
    "videos/roi_clip_scale_125.mp4",
    "videos/roi_clip_scale_100.mp4",
    "videos/roi_clip_scale_75.mp4",
    "videos/roi_clip_scale_50.mp4",
]

videos = [load_video_tensor(path) for path in video_paths]

min_frames = min(v.shape[0] for v in videos)
videos = [v[:min_frames] for v in videos]  # truncate to match

# Initialize the feature extractor
extractor = MultiScaleFeatureExtractor(pretrained=False)  # pretrained=True if you have internet

# Extract features
with torch.no_grad():
    features_by_scale = extractor(videos)

# Print feature shapes
for i, feat in enumerate(features_by_scale):
    print(f"Scale {i}: Feature shape = {feat.shape}")


feature_dim = features_by_scale[0].shape[1]  # usually 512
hidden_dim = 256
num_scales = len(features_by_scale)

# Create a fake decoder hidden state for testing
T = features_by_scale[0].shape[0]
decoder_hidden = torch.randn(T, hidden_dim)  # [T, H]

# Initialize the adaptive attention module
attn_module = AdaptiveAttention(
    feature_dim=feature_dim,
    hidden_dim=hidden_dim,
    num_scales=num_scales
)

# Apply attention-based fusion
fused_features, attn_weights = attn_module(features_by_scale, decoder_hidden)

# Print outputs
print(f"[RESULT] Fused features shape: {fused_features.shape}")      # Expected: [T, D]
print(f"[RESULT] Attention weights shape: {attn_weights.shape}")    # Expected: [T, num_scales]
print(f"[DEBUG] Attention weights at frame 0: {attn_weights[0]}")


temporal_encoder = BiLSTMTemporalEncoder(
    input_dim=feature_dim,
    hidden_dim=hidden_dim,
    num_layers=2,
    dropout=0.1
)

# Prepare fused features for BiLSTM
fused_features = fused_features.unsqueeze(0)  # [1, T, D]

# Forward through BiLSTM
temporal_output = temporal_encoder(fused_features)  # [1, T, 2*H]
print(f"[RESULT] Temporal encoder output shape: {temporal_output.shape}")

# === BERT-based Autoregressive Decoding ===
print("\n[INFO] Starting BERT-guided autoregressive decoding...")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

proj_to_bert = torch.nn.Linear(temporal_output.shape[-1], 768)
temporal_output_proj = proj_to_bert(temporal_output)  # [1, T, 768]

bert_decoder = TransformerDecoderWithContext(
    d_model=768,
    num_heads=8,
    num_layers=2,
    vocab_size=tokenizer.vocab_size
)

generated_tokens = [tokenizer.cls_token_id]
max_gen_len = 15

for step in range(max_gen_len):
    input_ids = torch.tensor([generated_tokens])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        bert_embeddings = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = bert_decoder(bert_embeddings, memory=temporal_output_proj)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()

    generated_tokens.append(next_token_id)
    if next_token_id == tokenizer.sep_token_id:
        break

decoded_sentence = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(f"[RESULT] BERT + Visual Context Predicted Sentence: {decoded_sentence}")