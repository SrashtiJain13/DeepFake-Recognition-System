from facenet_pytorch import MTCNN
from PIL import Image
import os

mtcnn = MTCNN()

BASE_DIR = "dataset/images"

for cls in ["real", "fake"]:
    folder = os.path.join(BASE_DIR, cls)
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        image = Image.open(path).convert("RGB")
        face = mtcnn(image)

        if face is None:
            print(f"❌ No face → Removing {path}")
            os.remove(path)

print("✅ Images without faces removed")
