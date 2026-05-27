import os
import random

BASE_DIR = "dataset/images"
REAL_DIR = os.path.join(BASE_DIR, "real")
FAKE_DIR = os.path.join(BASE_DIR, "fake")

SEED = 42
random.seed(SEED)

real_images = os.listdir(REAL_DIR)
fake_images = os.listdir(FAKE_DIR)

real_count = len(real_images)
fake_count = len(fake_images)

print("\n📊 BEFORE BALANCING")
print(f"REAL images : {real_count}")
print(f"FAKE images : {fake_count}")

min_count = min(real_count, fake_count)

random.shuffle(real_images)
random.shuffle(fake_images)


for img in real_images[min_count:]:
    os.remove(os.path.join(REAL_DIR, img))

for img in fake_images[min_count:]:
    os.remove(os.path.join(FAKE_DIR, img))


real_final = len(os.listdir(REAL_DIR))
fake_final = len(os.listdir(FAKE_DIR))

print("\n⚖️ AFTER BALANCING")
print(f"REAL images : {real_final}")
print(f"FAKE images : {fake_final}")

print("\n✅ Dataset successfully balanced")

