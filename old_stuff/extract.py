# ...existing code...
"""
Simple frame extractor: save N evenly spaced frames from a video into a folder.

Usage:
    Edit VIDEO_PATH, OUT_DIR, NUM_FRAMES, EXT and RESIZE below, then run:
    python extract.py

Dependencies:
    pip install opencv-python
"""
import os
import cv2
import math

# ------------------ User-editable variables ------------------
# Put the full path to your video file here:
VIDEO_PATH = r"C:\Users\marco\OneDrive\Documenti\speed-analysis\toextract\video10.mp4"

# Folder where extracted frames will be saved:
OUT_DIR = r"C:\Users\marco\OneDrive\Documenti\speed-analysis\imgs"

NAME="video10_"  # Used to create subfolder in OUT_DIR

# Number of frames to extract (evenly spaced):
NUM_FRAMES = 70

# Image extension: "jpg" or "png"
EXT = "jpg"

# Optional resize: set to (width, height) or None to keep original size
RESIZE = None  # e.g., (640, 360)
# --------------------------------------------------------------

def extract_frames(video_path: str, out_dir: str, count: int, ext: str = "jpg", resize: tuple = None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total == 0:
        cap.release()
        raise RuntimeError("Video contains no frames or cannot read frame count.")

    count = max(1, int(count))
    if count >= total:
        indices = list(range(total))
    else:
        if count == 1:
            indices = [total // 2]
        else:
            indices = [int(round(i * (total - 1) / (count - 1))) for i in range(count)]

    pad = len(str(len(indices)))
    saved = 0
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        out_name = f"{NAME}{i+1:0{pad}d}.{ext}"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, frame)
        saved += 1

    cap.release()
    return saved
 
if __name__ == "__main__":
    try:
        saved = extract_frames(VIDEO_PATH, OUT_DIR, NUM_FRAMES, ext=EXT, resize=RESIZE)
        print(f"Saved {saved} frames to {os.path.abspath(OUT_DIR)}")
    except Exception as e:
        print("Error:", e)
        raise 