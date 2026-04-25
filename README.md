# 🎯 Face Detection & Recognition (PyTorch + OpenCV)

A complete pipeline for **face detection, preprocessing, training, and evaluation** using **OpenCV** and **PyTorch**.

This project:

* Detects faces from raw images
* Preprocesses and organizes them
* Trains a custom CNN model
* Evaluates performance

🚧 **Planned Updates**

* Real-time webcam face detection
* Face recognition (identity prediction)
* Detection logging to file

---

## 📂 Project Structure

```
.
├── dataset/           # Raw images (organized by person)
├── faces/             # Extracted face images
├── detect.py          # Face detection & preprocessing
├── train.py           # Model training
├── test.py            # Model evaluation
├── face_model.pth     # Saved trained model
└── README.md
```

---

## ⚙️ Features

* Face detection using Haar Cascade
* Automatic dataset processing
* Largest face selection per image
* Image resizing (100x100)
* Data augmentation for robustness
* Custom CNN built from scratch
* Train/test split with reproducibility
* Accuracy and loss tracking

---

## 🧠 Pipeline Overview

### 1️⃣ Face Detection (`detect.py`)

* Reads images from `dataset/`
* Converts to grayscale
* Detects faces using OpenCV
* Keeps largest face
* Saves processed faces to `faces/`

---

### 2️⃣ Training (`train.py`)

* Loads processed dataset
* Applies augmentations:

  * Random horizontal flip
  * Rotation
  * Grayscale variation
* Trains CNN model
* Saves weights as `face_model.pth`

---

### 3️⃣ Testing (`test.py`)

* Loads trained model
* Evaluates on test split
* Outputs:

  * Accuracy
  * Loss

---

## 🧱 CNN Architecture

```
Input (3x100x100)
↓
Conv2D (16) + LeakyReLU + MaxPool
↓
Conv2D (32) + LeakyReLU + MaxPool
↓
Conv2D (64) + LeakyReLU + MaxPool
↓
Flatten
↓
Linear (128) + LeakyReLU
↓
Dropout (0.6)
↓
Output Layer (num_classes)
```

---

## 📦 Libraries Used

* Python 3.x
* opencv-python
* torch
* torchvision
* tqdm

---

## 🛠️ Installation

### 1. Clone Repository

```
git clone https://github.com/your-username/face-recognition.git
cd face-recognition
```

### 2. Install Dependencies

```
pip install opencv-python torch torchvision tqdm
```

👉 For GPU support, install PyTorch with CUDA:
https://pytorch.org/get-started/locally/

---

## ▶️ Usage

### Step 1: Prepare Dataset

```
dataset/
├── person1/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── person2/
│   ├── img1.jpg
│   ├── img2.jpg
```

---

### Step 2: Detect Faces

```
python detect.py
```

---

### Step 3: Train Model

```
python train.py
```

---

### Step 4: Test Model

```
python test.py
```

---

## 📊 Example Output

```
Epoch 10/50
Train Loss : 0.3456
Train Acc  : 89.25%
```

```
Test Accuracy: 85.40%
```

---

## 🚀 Future Improvements

* [ ] Real-time webcam detection
* [ ] Face recognition (embedding-based)
* [ ] Logging system (CSV / database)
* [ ] GUI interface
* [ ] Model optimization

---

## ⚠️ Notes

* Use clear, well-lit face images
* More images per person = better results
* Avoid extreme angles and occlusions

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Zaid Faraz
GitHub: https://github.com/your-username](https://github.com/zaidfaraz45)
