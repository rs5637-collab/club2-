import os
import sys
import zipfile
import random
import shutil
import subprocess
from pathlib import Path
import time

def install_if_missing(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        print(f"Installing {pkg} ...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--no-cache-dir", "--prefer-binary", pkg
        ])

# Install in safe order
install_if_missing("torch")
install_if_missing("torchvision")
install_if_missing("ultralytics")
install_if_missing("opencv-python-headless==4.10.0.84", "cv2")
install_if_missing("PyYAML", "yaml")
install_if_missing("pandas")

import torch
import cv2
from ultralytics import YOLO
import yaml
import pandas as pd

BASE = "/workspaces/club2-"  # Must match your folder name!
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

CLASSES = ["fall"]
DEVICE = "cpu"  # Force CPU — avoids hangs in Codespaces

for split in ["train", "validation"]:
    for sub in ["images", "labels"]:
        os.makedirs(os.path.join(DATA, split, sub), exist_ok=True)

ZIP_NAME = "data.zip"
zip_path = os.path.join(BASE, ZIP_NAME)

for _ in range(10):
    if os.path.exists(zip_path):
        break
    print(f"⏳ Waiting for {ZIP_NAME}... ({_ + 1}/10)")
    time.sleep(1)
else:
    sys.exit(f"❌ {ZIP_NAME} not found after 10 seconds. Please upload it to /workspaces/club2-")

print(f"📦 Found: {zip_path}")

extract_path = os.path.join(BASE, "_temp_extract")
os.makedirs(extract_path, exist_ok=True)

print("Extracting...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_path)
print("✅ Extracted.")

imgs, lbls = {}, {}
for root, _, files in os.walk(extract_path):
    for f in files:
        stem = Path(f).stem
        ext = Path(f).suffix.lower()
        full_path = os.path.join(root, f)
        if ext in [".jpg", ".jpeg", ".png"]:
            imgs[stem] = full_path
        elif ext == ".txt" and f != "classes.txt":
            lbls[stem] = full_path

pairs = [(imgs[k], lbls[k]) for k in imgs if k in lbls]
if not pairs:
    sys.exit("❌ No matching image-label pairs!")

print(f"📊 Found {len(pairs)} samples.")

random.seed(42)
random.shuffle(pairs)
cut = int(0.8 * len(pairs))
train_set, val_set = pairs[:cut], pairs[cut:]

def copy_file(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, Path(src).name))

for img, lbl in train_set:
    copy_file(img, os.path.join(DATA, "train/images"))
    copy_file(lbl, os.path.join(DATA, "train/labels"))

for img, lbl in val_set:
    copy_file(img, os.path.join(DATA, "validation/images"))
    copy_file(lbl, os.path.join(DATA, "validation/labels"))


data_yaml = {
    "train": os.path.abspath(os.path.join(DATA, "train/images")),
    "val": os.path.abspath(os.path.join(DATA, "validation/images")),
    "nc": len(CLASSES),
    "names": CLASSES
}
yaml_path = os.path.join(BASE, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f)
print("✅ data.yaml ready.")


print("🚀 Training on CPU (may take 2-5 mins)...")
model = YOLO("yolov8n.pt")  # Small model for speed

model.train(
    data=yaml_path,
    epochs=10,
    imgsz=224,
    batch=8,
    device=DEVICE,
    project=os.path.join(BASE, "runs"),
    name="train_fast",
    exist_ok=True,
    workers=2,
    hsv_h=0, hsv_s=0, hsv_v=0,
    degrees=0, translate=0, scale=0,
    shear=0, fliplr=0, mosaic=0, mixup=0
)


run_dir = os.path.join(BASE, "runs", "train_fast")
results_csv = os.path.join(run_dir, "results.csv")

if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    last = df.iloc[-1]
    map50 = last.get("metrics/mAP50(B)", float("nan"))
    print(f"\n🎯 Final mAP@0.5: {map50:.3f}")


out_zip = os.path.join(BASE, "fall_detection_model_fast.zip")
with zipfile.ZipFile(out_zip, "w") as zf:
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(best_pt):
        zf.write(best_pt, "best.pt")
    zf.write(yaml_path, "data.yaml")
    results_plot = os.path.join(run_dir, "results.png")
    if os.path.exists(results_plot):
        zf.write(results_plot, "training_results.png")

print(f"\n🎉 Done! Download: {out_zip}")
