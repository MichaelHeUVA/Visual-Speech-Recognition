# data/preprocess.py
import os
import glob
from tqdm import tqdm

def make_manifest(dataset_root, output_file, subset="train"):
    """
    Create a manifest file like:
    path/to/video.mp4|transcript
    """
    transcript_dir = os.path.join(dataset_root, "mvlrs_v1", "text")
    video_dir = os.path.join(dataset_root, "mvlrs_v1", "videos", subset)

    output_lines = []

    for txt_path in tqdm(sorted(glob.glob(f"{transcript_dir}/{subset}/*.txt"))):
        with open(txt_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name = parts[0]
            transcript = " ".join(parts[1:]).lower()
            video_path = os.path.join(video_dir, name + ".mp4")
            if os.path.exists(video_path):
                output_lines.append(f"{video_path}|{transcript}")

    with open(output_file, "w") as out:
        out.write("\n".join(output_lines))

    print(f"[INFO] Wrote {len(output_lines)} lines to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--subset", type=str, default="train", choices=["pretrain", "train", "val", "test"])
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    make_manifest(args.dataset_root, args.output_file, subset=args.subset)
