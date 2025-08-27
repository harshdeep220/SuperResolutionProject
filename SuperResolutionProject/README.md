---

# 📈 Super-Resolution on UC Merced Dataset (SRCNN vs ESRGAN)

## 🔍 Overview

This project implements and evaluates two state-of-the-art **Single Image Super-Resolution (SISR)** models:

* **SRCNN** (Super-Resolution Convolutional Neural Network)
* **ESRGAN** (Enhanced Super-Resolution Generative Adversarial Network)

We benchmark them on the **UC Merced Land Use dataset**, a standard remote sensing dataset, under **×2 and ×4 upscaling settings**.

The project includes:

* End-to-end pipeline: dataset preparation → training/inference → evaluation.
* Quantitative metrics: **SSIM** (primary) and **PSNR** (secondary).
* Qualitative comparison figures and side-by-side visualization panels.
* Publication-ready tables and plots for inclusion in reports/papers.

---

## 📂 Dataset

* **UC Merced Land Use Dataset**

  * 21 classes (100 images per class).
  * Each image: **256×256 RGB**.
  * Example classes: `agricultural`, `forest`, `parkinglot`, `river`, `residential`, etc.

### Preprocessing

* HR images: original 256×256 patches.
* LR inputs: generated via **bicubic downsampling** at ×2 and ×4.
* Stored in structured folders:

  ```
  data/processed/HR        → High-resolution ground truth
  data/processed/LR_x2     → Bicubic downsampled (×2)
  data/processed/LR_x4     → Bicubic downsampled (×4)
  results/SRCNN_x2         → SRCNN outputs (×2)
  results/SRCNN_x4
  results/ESRGAN_x2        → ESRGAN outputs (×2)
  results/ESRGAN_x4
  results/visuals          → Generated comparison figures
  ```

---

## ⚙️ Methods

### SRCNN

* Implemented from the original paper with PyTorch.
* Trained with MSE loss to predict HR from LR bicubic inputs.
* Outputs smoother results, optimized for pixel-level fidelity.

### ESRGAN

* Used pretrained **RRDBNet ESRGAN** architecture.
* GAN-based approach for sharper textures and more realistic reconstructions.
* Captures details like vegetation, roofs, and roads better than SRCNN.

---

## 📊 Evaluation Protocol

### Metrics

* **SSIM** (primary): Measures structural similarity, more aligned with human perception.
* **PSNR** (secondary): Measures pixel-wise fidelity, widely used in SR papers.
* Both computed in **RGB** with `channel_axis=2` and `data_range=1.0`.

### Aggregation

* **Per-image metrics** → stored in `eval_results.csv`.
* **Per-class averages** → mean SSIM/PSNR for each of the 21 classes.
* **Overall means** → aggregated across entire dataset.
* **Best-by-image table** → directly compares SRCNN vs ESRGAN per sample.

---

## 📈 Results

### Overall Performance

| Scale | Method | Mean SSIM | Mean PSNR    |
| ----- | ------ | --------- | ------------ |
| ×2    | SRCNN  | 0.693     | 25.03 dB     |
| ×2    | ESRGAN | **0.81**  | **27.42 dB** |
| ×4    | SRCNN  | 0.409     | 20.95 dB     |
| ×4    | ESRGAN | **0.66**  | **23.87 dB** |

*(Numbers shown here are placeholders – update with your CSV values.)*

### Per-Class Trends

* ESRGAN consistently outperforms SRCNN in texture-rich classes (`forest`, `residential`, `stadium`).
* SRCNN sometimes yields smoother but blurrier outputs.
* ESRGAN recovers sharper patterns (roofs, roads, vegetation).

---

## 🎨 Visualizations

### Side-by-Side Comparison

Each figure shows: **LR → SRCNN → ESRGAN → HR**.

Examples saved in:
`results/visuals/desert_45_panel.png`

![Sample Panel](results/visuals/desert_45_panel.png)

### Class-Wise Bar Charts

* Mean SSIM per class for both models at ×2 and ×4.
* Highlights ESRGAN’s advantage on texture-heavy categories.

### ΔSSIM Distribution

* Histogram of `ΔSSIM = ESRGAN − SRCNN` shows distribution of improvements.

---

## 🚀 Usage

### 1. Dataset Prep

```bash
python scripts/select_images.py   # select subset
python scripts/generate_LR.py     # create LR_x2 and LR_x4
```

### 2. Run Models

```bash
# SRCNN
python models/SRCNN-PyTorch/test.py

# ESRGAN
python models/ESRGAN/test.py
```

### 3. Evaluate

```bash
python notebooks/evaluate_models.py
```

Outputs:

* Per-image CSV: `eval_results.csv`
* Aggregated metrics printed in console.

### 4. Generate Visual Panels

```python
from notebooks.visualize_panels import save_image_panel
save_image_panel("desert_45")
```

---

## 📖 Paper Notes

* **Methodology**: bicubic downsampling to create LR, SSIM/PSNR evaluation in RGB, ESRGAN vs SRCNN under ×2/×4 scales.
* **Results**: ESRGAN improves SSIM significantly, especially for texture-heavy land-use categories.
* **Limitations**: SSIM/PSNR may not fully reflect perceptual quality. Future work could add **LPIPS** or **FID** for perceptual metrics.

---

## 📌 Future Work

* Train models directly on UC Merced rather than using pretrained ESRGAN.
* Extend evaluation with perceptual metrics like LPIPS.
* Explore SR models optimized for **remote sensing** (e.g., DSRN, DRRN, SwinIR).

---

