import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
from facenet_pytorch import MTCNN


DATASET_DIR = "dataset/images"
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
IMAGE_SIZE = 224
MODEL_SAVE_PATH = "image_model_effnet.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- FACE DETECTOR ----------------
mtcnn = MTCNN(image_size=IMAGE_SIZE, margin=20)

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- DATASET ----------------
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.samples = []
        self.transform = transform

        for label, folder in enumerate(["real", "fake"]):
            folder_path = os.path.join(root_dir, folder)
            for img in os.listdir(folder_path):
                self.samples.append((os.path.join(folder_path, img), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        face = mtcnn(img)
        if face is None:
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = self.transform(img)
        else:
            img = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(face)

        return img, label

# ---------------- SPLIT DATA ----------------
full_dataset = DeepfakeDataset(DATASET_DIR, train_transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=2
)
model.to(device)

class_weights = torch.tensor([1.0, 2.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("\n🔥 TRAINING STARTED 🔥\n")

for epoch in range(EPOCHS):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = 100 * train_correct / train_total

    # ---- VALIDATION ----
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}] "
        f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%\n"
    )

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved as {MODEL_SAVE_PATH}")

