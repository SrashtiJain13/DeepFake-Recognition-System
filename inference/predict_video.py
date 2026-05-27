from pathlib import Path
import sys

import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.video_model import DeepFakeVideoModel


IMAGE_SIZE = 224
NUM_FRAMES = 20
FAKE_CLASS_INDEX = 1
REAL_CLASS_INDEX = 0
CONFIDENCE_MARGIN_PERCENT = 5.0
MIN_FACE_FRAMES_FOR_CONFIDENT_VIDEO = 8
MIN_REAL_CONFIDENCE_PERCENT = 65.0
MIN_FAKE_CONFIDENCE_PERCENT = 60.0
MODEL_PATH = PROJECT_ROOT / "video_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_mtcnn = None


def _get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Video model not found at: {MODEL_PATH}")

        _model = DeepFakeVideoModel().to(device)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        _model.load_state_dict(state_dict)
        _model.eval()

    return _model


def _get_mtcnn():
    global _mtcnn

    if _mtcnn is None:
        _mtcnn = MTCNN(
            image_size=IMAGE_SIZE,
            margin=20,
            device=device,
        )

    return _mtcnn


def extract_frames(video_path, max_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video has no frames")

    frames = []
    target_count = min(max_frames, total_frames)
    target_indices = {
        int(idx.item())
        for idx in torch.linspace(0, total_frames - 1, steps=target_count)
    }
    target_indices.add(total_frames - 1)

    for frame_index in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index not in target_indices:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if len(frames) >= target_count:
            break

    cap.release()

    if not frames:
        raise ValueError("No frames extracted from video")

    return frames


def _select_stable_frames(face_tensors, target_count=NUM_FRAMES):
    if not face_tensors:
        return []

    if len(face_tensors) == 1:
        return [face_tensors[0].clone() for _ in range(target_count)]

    if len(face_tensors) >= target_count:
        
        center_weight = torch.linspace(0.85, 1.15, steps=len(face_tensors))
        weighted_positions = torch.cumsum(center_weight, dim=0)
        sample_points = torch.linspace(
            weighted_positions[0],
            weighted_positions[-1],
            steps=target_count,
        )
        indices = torch.searchsorted(weighted_positions, sample_points).clamp(max=len(face_tensors) - 1)
        return [face_tensors[int(idx.item())] for idx in indices]

    selected_frames = list(face_tensors)
    while len(selected_frames) < target_count:
        mirrored_index = len(selected_frames) % len(face_tensors)
        selected_frames.append(face_tensors[mirrored_index].clone())
    return selected_frames


def _to_face_tensor(frame_rgb):
    face_tensor = _get_mtcnn()(Image.fromarray(frame_rgb))
    if face_tensor is None:
        return None
    return face_tensor.cpu()


def predict_video(video_path):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames = extract_frames(video_path)

    detected_face_frames = []

    for frame in frames:
        face_tensor = _to_face_tensor(frame)
        if face_tensor is not None:
            detected_face_frames.append(face_tensor)

    if not detected_face_frames:
        return {
            "label": "NO FACE DETECTED",
            "real_prob": 0.0,
            "fake_prob": 0.0,
        }

    
    selected_frames = _select_stable_frames(detected_face_frames, target_count=NUM_FRAMES)

    video_tensor = torch.stack(selected_frames).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = _get_model()(video_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)

    real_prob = probs[REAL_CLASS_INDEX].item() * 100.0
    fake_prob = probs[FAKE_CLASS_INDEX].item() * 100.0
    confidence_gap = abs(fake_prob - real_prob)

    if len(detected_face_frames) < MIN_FACE_FRAMES_FOR_CONFIDENT_VIDEO:
        prediction = "UNCERTAIN"
        decision_reason = "Too few face-detected frames were available for a reliable result."
    elif confidence_gap < CONFIDENCE_MARGIN_PERCENT:
        prediction = "UNCERTAIN"
        decision_reason = "The real and fake scores are too close together."
    elif fake_prob > real_prob and fake_prob >= MIN_FAKE_CONFIDENCE_PERCENT:
        prediction = "FAKE"
        decision_reason = "The fake score is clearly stronger than the real score."
    elif real_prob > fake_prob and real_prob >= MIN_REAL_CONFIDENCE_PERCENT:
        prediction = "REAL"
        decision_reason = "The real score is strong enough and clearly above the fake score."
    else:
        prediction = "UNCERTAIN"
        decision_reason = "The leading score is not strong enough for a confident final label."

    return {
        "label": prediction,
        "real_prob": round(real_prob, 2),
        "fake_prob": round(fake_prob, 2),
        "confidence_gap": round(confidence_gap, 2),
        "confidence_threshold": CONFIDENCE_MARGIN_PERCENT,
        "face_frames_used": len(detected_face_frames),
        "sampled_frames": len(frames),
        "decision_reason": decision_reason,
        "min_real_confidence": MIN_REAL_CONFIDENCE_PERCENT,
        "min_fake_confidence": MIN_FAKE_CONFIDENCE_PERCENT,
    }


if __name__ == "__main__":
    user_video_path = input("Enter video path: ").strip()

    try:
        result = predict_video(user_video_path)
        print("\nPrediction Result")
        print("-----------------")
        print(f"Prediction: {result['label']}")
        print(f"Real: {result['real_prob']:.2f}%")
        print(f"Fake: {result['fake_prob']:.2f}%")
    except Exception as exc:
        print(f"Error: {exc}")
