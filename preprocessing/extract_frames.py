import cv2
import os
from tqdm import tqdm

VIDEO_DIR = "dataset/videos"
FRAME_DIR = "dataset/video_frames"

os.makedirs(FRAME_DIR, exist_ok=True)

valid_ext = (".mp4", ".avi", ".mov", ".mkv")

for label in ["real", "fake"]:

    input_path = os.path.join(VIDEO_DIR, label)
    output_path = os.path.join(FRAME_DIR, label)

    os.makedirs(output_path, exist_ok=True)

    videos = [v for v in os.listdir(input_path) if v.lower().endswith(valid_ext)]

    for video in tqdm(videos, desc=f"Processing {label}"):

        video_path = os.path.join(input_path, video)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Cannot open {video}")
            continue

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Take every 10th frame
            if frame_count % 10 == 0:

                # Convert BGR → RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize for training efficiency
                frame = cv2.resize(frame, (224, 224))

                name = f"{os.path.splitext(video)[0]}_f{frame_count}.jpg"

                cv2.imwrite(os.path.join(output_path, name), frame)
                saved_count += 1

            frame_count += 1

        cap.release()

        if saved_count == 0:
            print(f"⚠️ No frames saved from {video}")