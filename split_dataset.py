import os
import shutil
import random


BASE_DIR = "dataset/images"
OUTPUT_DIR = "dataset/images_split"

CLASSES = ["real", "fake"]
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)

for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)


def split_class(class_name):
    src_dir = os.path.join(BASE_DIR, class_name)
    images = os.listdir(src_dir)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, imgs in splits.items():
        for img in imgs:
            src = os.path.join(src_dir, img)
            dst = os.path.join(OUTPUT_DIR, split, class_name, img)
            shutil.copy(src, dst)

    print(f"✅ {class_name.upper()} → Train:{len(splits['train'])} "
          f"Val:{len(splits['val'])} Test:{len(splits['test'])}")


print("\n📁 Splitting dataset...\n")
for cls in CLASSES:
    split_class(cls)

print("\n🎯 Dataset split completed successfully!")