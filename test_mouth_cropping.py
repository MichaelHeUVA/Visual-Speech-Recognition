from detector import LandmarksDetector
import cv2
import torchvision
from data_loader import AVSRDataLoader


def save2vid(filename, vid, frames_per_second):
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.io.write_video(filename, vid, int(frames_per_second))


def preprocess_video(src_filename, dst_filename):
    landmarks = landmarks_detector(src_filename)
    data = dataloader.load_data(src_filename, landmarks)
    fps = cv2.VideoCapture(src_filename).get(cv2.CAP_PROP_FPS)
    save2vid(dst_filename, data, fps)
    return


dataloader = AVSRDataLoader(
    speed_rate=1,
    convert_gray=False,
)
landmarks_detector = LandmarksDetector()

preprocess_video(src_filename="videos/doctor.mp4", dst_filename="videos/roi_doctor.mp4")
