import os
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

mtcnn = MTCNN(image_size=224, margin=20)

INPUT_DIR = "dataset/images"
OUTPUT_DIR = "dataset/processed_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in ["real", "fake"]:
    os.makedirs(f"{OUTPUT_DIR}/{label}", exist_ok=True)
    for img_name in tqdm(os.listdir(f"{INPUT_DIR}/{label}")):
        img_path = f"{INPUT_DIR}/{label}/{img_name}"
        try:
            img = Image.open(img_path).convert("RGB")
            face = mtcnn(img)
            if face is not None:
                face_img = face.permute(1,2,0).numpy()
                face_img = (face_img * 255).astype("uint8")
                cv2.imwrite(f"{OUTPUT_DIR}/{label}/{img_name}", face_img)
        except:
            continue
