print("Started")
import os
from PIL import Image

BASE_DIR = "dataset/images"

def clean_folder(folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            with Image.open(path) as img:
                img.verify()  
        except Exception:
            print(f"Removing corrupt file: {path}")
            try:
                os.remove(path)
            except PermissionError:
                print(f"Skipped (file in use): {path}")

for cls in ["real", "fake"]:
    clean_folder(os.path.join(BASE_DIR, cls))

print("✅ Corrupt image cleaning completed")

