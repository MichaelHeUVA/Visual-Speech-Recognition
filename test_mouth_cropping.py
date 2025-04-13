from detector import LandmarksDetector
import cv2
import torchvision
from data_loader import AVSRDataLoader


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
    scales=[1.5, 1.25, 1.0, 0.75,0.5]
)

# preprocess_video(src_filename="videos/doctor.mp4", dst_filename="videos/roi_doctor.mp4")
