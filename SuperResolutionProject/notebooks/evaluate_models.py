import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# === Paths ===
HR_DIR   = "/content/drive/MyDrive/SuperResolutionProject/data/processed/HR"
SRCNN_X2 = "/content/drive/MyDrive/SuperResolutionProject/results/SRCNN_x2"
SRCNN_X4 = "/content/drive/MyDrive/SuperResolutionProject/results/SRCNN_x4"
ESRGAN_X2 = "/content/drive/MyDrive/SuperResolutionProject/results/ESRGAN_x2"
ESRGAN_X4 = "/content/drive/MyDrive/SuperResolutionProject/results/ESRGAN_x4"

methods = {
    "SRCNN": {2: SRCNN_X2, 4: SRCNN_X4},
    "ESRGAN": {2: ESRGAN_X2, 4: ESRGAN_X4}
}

# === Helpers ===
def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = []

# === Evaluation Loop ===
for method, scales in methods.items():
    for scale, pred_dir in scales.items():
        for fname in os.listdir(pred_dir):
            if not fname.lower().endswith(".png"):
                continue

            stem = os.path.splitext(fname)[0]
            # Normalize stem for fair HR matching
            for suffix in ["_x2", "_x4", "_rlt", "_ESRGAN", "_SRCNN", "_out"]:
              stem = stem.replace(suffix, "")

            hr_path = os.path.join(HR_DIR, stem + ".png")
            pred_path = os.path.join(pred_dir, fname)

            if not os.path.exists(hr_path):
                print(f"⚠️ Skipping {fname}, no HR match found")
                continue

            hr = read_img(hr_path)
            pr = read_img(pred_path)
            if hr is None or pr is None:
                continue

            # Ensure same size
            if hr.shape != pr.shape:
                pr = cv2.resize(pr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

            # Normalize to [0,1]
            hr_f = hr.astype(np.float32) / 255.0
            pr_f = pr.astype(np.float32) / 255.0

            ssim_val = ssim(hr_f, pr_f, channel_axis=2, data_range=1.0)
            psnr_val = psnr(hr_f, pr_f, data_range=1.0)

            results.append({
                "class": stem.split("_")[0],
                "image": stem,
                "scale": f"x{scale}",
                "method": method,
                "SSIM": ssim_val,
                "PSNR": psnr_val
            })

# === Save Results ===
df = pd.DataFrame(results)
out_csv = "/content/drive/MyDrive/SuperResolutionProject/eval_results.csv"
df.to_csv(out_csv, index=False)
print(f"✅ Saved per-image results to {out_csv}")

# Debug: show first few rows & columns
print("\n🔍 DataFrame preview:")
print(df.head())
print("Columns:", df.columns.tolist())

# === Aggregate Safely ===
if "scale" in df.columns and "method" in df.columns:
    print("\n📊 Overall Means:")
    print(df.groupby(["scale","method"])[["SSIM","PSNR"]].mean())

    print("\n📊 Per-Class Means:")
    print(df.groupby(["scale","method","class"])[["SSIM","PSNR"]].mean())
else:
    print("⚠️ 'scale' or 'method' column missing in DataFrame. Please check naming logic.")
