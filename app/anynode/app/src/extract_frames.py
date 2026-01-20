# tools/extract_frames.py
import os, cv2

def extract_every_n_seconds(video_path, out_dir, n=1.0, max_frames=300):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * n)))
    i = 0; saved = 0
    while True:
        ret, frame = cap.read()
        if not ret or saved >= max_frames: break
        if i % step == 0:
            out = os.path.join(out_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(out, frame)
            saved += 1
        i += 1
    cap.release()
    return saved
