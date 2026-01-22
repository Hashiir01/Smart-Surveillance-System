# 🚨 Smart Surveillance System

A deep learning-based surveillance system designed to detect **fighting (violent behavior)** and **smoking** in videos, images, or live camera streams. This solution integrates two independently trained CNN models and offers a modern user interface via Streamlit.

---

## 🧠 Project Summary

The system consists of:
- A **violence detection model** trained on frames extracted from labeled video clips.
- A **smoking detection model** trained on labeled smoking images.
- A **Streamlit web app** (`temp_app.py`) for real-time and batch predictions using webcam, image, or video input.

---

## 🗂️ Directory Structure

```

SmartSurveillanceSystem/
│
├── Dataset/
│   ├── Violence/                      # Raw videos with violent activity
│   ├── NonViolence/                  # Raw videos without violence
│   └── frames/                       # Extracted image frames for violence classification
│       ├── violence/
│       └── non\_violence/
│
├── Scripts/
│   ├── frame\_extractor.py            # Extract frames from videos
│   ├── data\_preprocessing.py         # Data loading and transformation logic
│   ├── model.py                      # ViolenceClassifier (ResNet18-based)
│   ├── train\_initial.py              # Initial training for violence detection
│   ├── train\_update.py               # Fine-tune existing violence model
│   ├── model\_train.py                # Train smoking detection model
│   ├── evaluate.py                   # Evaluate violence model performance
│   ├── temp\_app.py                   # Streamlit application for inference
│   └── Saved Model/
│       ├── model\_checkpoint.pth          # Trained weights for violence model
│       └── smoking\_model\_weights.pt      # Trained weights for smoking model

````

---

## ⚙️ Setup Instructions

### 1. Install Required Libraries

```bash
pip install torch torchvision opencv-python pillow streamlit scikit-learn
````

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

### 2. Extract Frames from Videos

Run to convert your video dataset into frames:

```bash
python Scripts/frame_extractor.py
```

This saves labeled image frames to `Dataset/frames/` for training the violence detection model.

---

## 🔧 Model Training

### A. Train Violence Detection Model

```bash
python Scripts/train_initial.py
```

* Loads extracted frames from `Dataset/frames/`
* Trains a `ResNet18`-based classifier
* Saves the model to `Saved Model/model_checkpoint.pth`

### B. Update Existing Violence Model

```bash
python Scripts/train_update.py
```

* Loads previous checkpoint and continues training

### C. Train Smoking Detection Model

```bash
python Scripts/model_train.py
```

* Loads images from `dataset/` directory (smoking vs non-smoking)
* Trains a binary classifier
* Saves to `Saved Model/smoking_model_weights.pt`

---

## ✅ Model Evaluation

To test the violence model:

```bash
python Scripts/evaluate.py
```

Displays:

* Accuracy
* Confusion matrix

---

## 🖥️ Run the Streamlit App

```bash
streamlit run Scripts/temp_app.py
```

### Features:

* 🔍 **Upload Image** — Get instant predictions
* 📼 **Upload Video** — Frame-by-frame detection
* 🎥 **Live Camera** — Real-time detection via webcam

Choose between **Smoking Detection** or **Violence Detection** from the sidebar.

---

## 📊 Model Architecture

Both models are based on **ResNet18**:

* Smoking: output layer → `nn.Linear(..., 1)` with `BCEWithLogitsLoss`
* Violence: output layer → `nn.Linear(..., 2)` with `CrossEntropyLoss`

---

## 👨‍💻 Contributors

| Name               | Role                    |
| ------------------ | ----------------------- |
| 👩‍💻 Adan Akbar   | ML Modeling             |
| 👨‍💻 Bilal Qaisar | Streamlit + Integration |
| 👨‍💻 Hashir Nasir | Video Processing Logic  |

---

## 📌 Future Enhancements

* Add object detection (e.g., YOLO) for region-based detection
* Send alerts upon detection
* Extend model to multi-label classification

---

## 📜 License

This project is for academic and research purposes only.
