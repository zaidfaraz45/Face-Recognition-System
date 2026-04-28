# 🎯 Face Recognition & Attendance System
### PyTorch + OpenCV

A complete end-to-end pipeline for **face detection, deep learning training, and real-time biometric attendance** using **OpenCV** and **PyTorch**.

This system has been optimized to handle **18 identities** (17 specific individuals + 1 "Unknown" class) with a focus on high generalization, bridging the gap between training environments and real-world deployment.

---

## 🚀 Performance Metrics

| Metric | Value |
|---|---|
| Training Accuracy | ~97% |
| Testing Accuracy | ~83% |
| Train/Test Gap | ~14% (reduced from ~20%) |

> **Optimization:** Overfitting was significantly reduced using L2 Regularization (Weight Decay), high Dropout (0.7), and aggressive Data Augmentation.

---

## 📂 Project Structure

```
.
├── dataset/           # Raw images (organized by person)
├── faces/             # Extracted & aligned face crops (100×100)
├── attendance/        # Auto-generated CSV attendance logs
├── detect.py          # Face detection & preprocessing (Haar Cascades)
├── model.py           # Deep 4-Layer CNN Architecture
├── train.py           # Model training with Data Augmentation
├── test.py            # Independent model evaluation script
├── camera.py          # Real-time recognition & attendance logger
├── face_model.pth     # Saved trained weights
└── README.md
```

---

## ⚙️ Key Features

- **Haar Cascade Integration** — Fast face localization with optimized 5% padding for feature isolation.
- **4-Layer CNN Architecture** — Deep feature extraction using BatchNorm, LeakyReLU, and high Dropout (0.7) to prevent memorization.
- **Smart Attendance Logging** — Real-time recognition with live CSV logging and duplication prevention (logs each person once per day).
- **Robust Training** — Implements Label Smoothing (0.05) and Weight Decay (1e-2) to ensure the model generalizes across new lighting conditions and angles.
- **Dynamic Data Augmentation** — Real-time synthesis of sharpness, rotation, and color shifts during training.

---

## 🧠 Pipeline Overview

### 1️⃣ Preprocessing — `detect.py`

- Scans `dataset/` and converts raw images to grayscale.
- Detects faces and isolates the largest face per image.
- Crops and resizes to **100×100** resolution to focus the CNN on internal facial geometry rather than background pixels.

### 2️⃣ Training — `train.py`

- **Augmentations:** Horizontal flips, 15° rotations, color jitter, and sharpness adjustments.
- **Loss:** `CrossEntropyLoss` with label smoothing to discourage over-confident incorrect predictions.
- **Scheduler:** `ReduceLROnPlateau` for precise weight convergence once the loss curve stalls.

### 3️⃣ Real-Time Inference — `camera.py`

- Live webcam feed processing with identity mapping.
- **Thresholding:** Requires a softmax confidence of ≥ 80% for identity verification.
- **Visual Feedback:** 🟢 Green bounding boxes for verified users, 🔴 Red for "Unknown" or low-confidence detections.

---

## 🧱 CNN Architecture

```
Input (3 × 100 × 100)
        ↓
Conv Block 1 — 32 filters  | BatchNorm + LeakyReLU + MaxPool
        ↓
Conv Block 2 — 64 filters  | BatchNorm + LeakyReLU + MaxPool
        ↓
Conv Block 3 — 128 filters | BatchNorm + LeakyReLU + MaxPool
        ↓
Conv Block 4 — 256 filters | BatchNorm + LeakyReLU + MaxPool
        ↓
AdaptiveAvgPool (4 × 4)
        ↓
Fully Connected — 1024 nodes | LeakyReLU + Dropout (0.7)
        ↓
Fully Connected — 512 nodes  | LeakyReLU
        ↓
Output Layer — 18 Classes
```

---

## 🛠️ Installation & Usage

### 1. Install Dependencies

```bash
pip install opencv-python torch torchvision tqdm pillow
```

### 2. Run the Pipeline

| Step | Command | Description |
|---|---|---|
| Prepare | *(manual)* | Place raw images in `dataset/person_name/` |
| Detect | `python detect.py` | Extract and save face crops |
| Train | `python train.py` | Train the CNN model |
| Evaluate | `python test.py` | Run evaluation on the test set |
| Deploy | `python camera.py` | Start real-time attendance system |

---

## 📦 Libraries Used

| Library | Purpose |
|---|---|
| OpenCV | Face detection and image processing |
| PyTorch | Neural network training and inference |
| Torchvision | Data transformations and datasets |
| Tqdm | Training progress visualization |
| Pillow | Image loading and format handling |

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

## 👨‍💻 Author

**Zaid Faraz**  
GitHub: [@zaidfaraz45](https://github.com/zaidfaraz45)