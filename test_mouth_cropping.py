import cv2
import torch
import torchvision
from torchvision import transforms
from data_loader import AVSRDataLoader
from feature_extractor import MultiScaleFeatureExtractor
from detector import LandmarksDetector

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
    src_filename="videos/doctor.mp4",
    dst_prefix="videos/roi_doctor",
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
    "videos/roi_doctor_scale_150.mp4",
    "videos/roi_doctor_scale_125.mp4",
    "videos/roi_doctor_scale_100.mp4",
    "videos/roi_doctor_scale_75.mp4",
    "videos/roi_doctor_scale_50.mp4",
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