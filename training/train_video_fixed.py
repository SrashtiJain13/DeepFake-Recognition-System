from pathlib import Path
import re
import sys

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.video_model import DeepFakeVideoModel
from utils import get_device, validate_numpy_torch_compatibility


VIDEO_DIR = ROOT_DIR / "dataset" / "video_frames"
MODEL_SAVE_PATH = ROOT_DIR / "video_model.pth"
NUM_FRAMES = 20
IMAGE_SIZE = 224
BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-4
TRAIN_SPLIT = 0.8
FRAME_PATTERN = re.compile(r"^(?P<video_id>.+)_f(?P<frame_idx>\d+)\.jpg$")

device = get_device()
validate_numpy_torch_compatibility()

mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=20,
    device=device,
)


class VideoFrameDataset(Dataset):
    def __init__(self, root_dir: Path):
        self.samples = self._build_samples(root_dir)

    def _build_samples(self, root_dir: Path):
        samples = []

        for label, cls_name in enumerate(["real", "fake"]):
            cls_dir = root_dir / cls_name
            grouped_frames = {}

            for frame_path in cls_dir.glob("*.jpg"):
                match = FRAME_PATTERN.match(frame_path.name)
                if match is None:
                    continue

                video_id = match.group("video_id")
                frame_idx = int(match.group("frame_idx"))
                grouped_frames.setdefault(video_id, []).append((frame_idx, frame_path))

            for video_id, frames in grouped_frames.items():
                frames.sort(key=lambda item: item[0])
                ordered_paths = [frame_path for _, frame_path in frames]
                samples.append((video_id, ordered_paths, label))

        if not samples:
            raise FileNotFoundError(f"No frame sequences found in: {root_dir}")

        return samples

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, frame_paths):
        if len(frame_paths) >= NUM_FRAMES:
            indices = torch.linspace(0, len(frame_paths) - 1, steps=NUM_FRAMES)
            return [frame_paths[int(idx.item())] for idx in indices]

        padded = list(frame_paths)
        while len(padded) < NUM_FRAMES:
            padded.append(padded[-1])
        return padded

    def _load_face_tensor(self, frame_path: Path):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(Image.fromarray(frame))

        if face is None:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

        return face.cpu()

    def __getitem__(self, idx):
        _, frame_paths, label = self.samples[idx]
        selected_frames = self._sample_frames(frame_paths)
        faces = [self._load_face_tensor(frame_path) for frame_path in selected_frames]
        video_tensor = torch.stack(faces)
        return video_tensor, torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def main():
    dataset = VideoFrameDataset(VIDEO_DIR)
    print(f"Dataset size: {len(dataset)} sequences")

    train_size = max(1, int(len(dataset) * TRAIN_SPLIT))
    val_size = len(dataset) - train_size

    if val_size == 0 and len(dataset) > 1:
        train_size -= 1
        val_size = 1

    if val_size == 0:
        raise ValueError("Need at least 2 video sequences to create train/validation splits.")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DeepFakeVideoModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\nVIDEO TRAINING STARTED\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / max(len(train_loader), 1)
        train_acc = 100.0 * total_correct / max(total_samples, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.2f}%"
        )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Video model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
