import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
from facenet_pytorch import MTCNN


TEST_DIR = "dataset/images"
MODEL_PATH = "image_model_effnet.pth"
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(image_size=IMAGE_SIZE, margin=20)


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

y_true = []
y_pred = []

print("\n🧪 IMAGE TESTING STARTED...\n")

for label, folder in enumerate(["real", "fake"]):
    folder_path = os.path.join(TEST_DIR, folder)

    for img_name in tqdm(os.listdir(folder_path), desc=f"Testing {folder.upper()}"):
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path).convert("RGB")

        face = mtcnn(img)

        if face is None:
            img = transform(img).unsqueeze(0).to(device)
        else:
            img = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(face).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            _, pred = torch.max(outputs, 1)

        y_true.append(label)
        y_pred.append(pred.item())

TP = sum((p == 1 and t == 1) for p, t in zip(y_pred, y_true))
TN = sum((p == 0 and t == 0) for p, t in zip(y_pred, y_true))
FP = sum((p == 1 and t == 0) for p, t in zip(y_pred, y_true))
FN = sum((p == 0 and t == 1) for p, t in zip(y_pred, y_true))

accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
precision = TP / (TP + FP + 1e-8)
recall = TP / (TP + FN + 1e-8)
f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

print("\n==============================")
print(f"✅ Accuracy: {accuracy*100:.2f}%")
print(f"🎯 Precision: {precision:.4f}")
print(f"🔍 Recall: {recall:.4f}")
print(f"⚖️ F1 Score: {f1:.4f}")
print("==============================\n")