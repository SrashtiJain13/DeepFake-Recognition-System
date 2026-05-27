import cv2
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()


def validate_numpy_torch_compatibility():
    try:
        torch.tensor([0]).numpy()
    except Exception as exc:
        numpy_version = getattr(np, "__version__", "unknown")
        torch_version = getattr(torch, "__version__", "unknown")
        raise RuntimeError(
            "NumPy is installed but is not compatible with the current PyTorch build. "
            f"Detected torch=={torch_version} and numpy=={numpy_version}. "
            "This usually happens when PyTorch was built against NumPy 1.x but NumPy 2.x is installed. "
            "Use the project's virtual environment or reinstall with 'numpy<2'."
        ) from exc

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_image_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )
    return model

def extract_face(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    if len(faces) == 0:
        return None

    # take largest face
    x, y, w, h = sorted(
        faces, key=lambda f: f[2] * f[3], reverse=True
    )[0]

    face = img[y:y+h, x:x+w]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    return Image.fromarray(face)

def preprocess_image(image_path):
    face = extract_face(image_path)

    if face is None:
        return None

    image_tensor = image_transform(face)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    return image_tensor

def extract_frames(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // max_frames, 1)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            continue

        frames.append(frame)

        if len(frames) >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        return None

    return frames
