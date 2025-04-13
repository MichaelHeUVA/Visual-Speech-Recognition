import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import os

class LRS2Dataset(Dataset):
    def __init__(self, manifest_file, tokenizer, transform=None, max_len=100):
        self.samples = []
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

        with open(manifest_file, "r") as f:
            for line in f:
                path, text = line.strip().split("|", 1)
                self.samples.append((path, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, text = self.samples[idx]

        frames, _, _ = read_video(video_path, pts_unit="sec")  # [T, H, W, C]
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]

        if self.transform:
            frames = self.transform(frames)

        # Tokenize transcript
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)

        return frames, input_ids, attention_mask
