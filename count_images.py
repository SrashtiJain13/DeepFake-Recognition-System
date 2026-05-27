import os

BASE_DIR = "dataset/images"

for cls in ["real", "fake"]:
    count = len(os.listdir(os.path.join(BASE_DIR, cls)))
    print(f"{cls.upper()} images: {count}")
