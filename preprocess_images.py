import cv2
import os

INPUT_DIR = "dataset/train"
OUTPUT_DIR = "dataset/processed/train"

IMG_SIZE = 299  # required for Xception later

os.makedirs(os.path.join(OUTPUT_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "fake"), exist_ok=True)

def process_images(label):
    in_path = os.path.join(INPUT_DIR, label)
    out_path = os.path.join(OUTPUT_DIR, label)

    for img_name in os.listdir(in_path):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(in_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # normalize

        save_path = os.path.join(out_path, img_name)
        cv2.imwrite(save_path, (img * 255).astype("uint8"))

process_images("real")
process_images("fake")

print("Preprocessing complete")

