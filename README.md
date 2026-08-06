# 🛒 Retail Shelf Detection System

AI-powered retail shelf analysis system for **product detection, classification, inventory auditing, and planogram compliance** using **YOLOv8, RT-DETR, ResNet50, and Streamlit**.

---

## 📸 Demo

### Home Interface

<p align="center">
<img src="results/screenshots/Screenshot%202026-08-07%20022210.png" width="900">
</p>

---

### Audit Dashboard

<p align="center">
<img src="results/screenshots/Screenshot%202026-08-07%20022221.png" width="900">
</p>

---

### Analytics

<p align="center">
<img src="results/screenshots/Screenshot%202026-08-07%20022231.png" width="900">
</p>

---

## 📖 Overview

This project provides an end-to-end intelligent retail shelf analysis pipeline capable of:

- Detecting products on retail shelves
- Classifying detected products
- Comparing multiple object detection models
- Auditing inventory automatically
- Running inference through an interactive Streamlit application

The system was developed as an academic Computer Vision project focusing on dense object detection in retail environments.

---

## ✨ Features

- Real-time product detection
- YOLOv8 detector
- RT-DETR detector
- ResNet50 product classification
- Interactive Streamlit interface
- Inventory dashboard
- SKU statistics
- CSV report export
- Visual prediction outputs
- Model comparison

---

## 📊 Model Performance

| Metric | YOLOv8 | RT-DETR |
|--------|--------:|--------:|
| mAP@0.5 | **0.9359** | 0.9029 |
| mAP@0.5:0.95 | **0.6185** | 0.5933 |
| Precision | **0.9079** | 0.8793 |
| Recall | **0.8828** | 0.8702 |
| F1 Score | **0.8952** | 0.8748 |
| Inference Time | **38 ms** | 70 ms |

### Highlights

- 🥇 YOLOv8 achieved the highest detection accuracy.
- ⚡ YOLOv8 was nearly twice as fast as RT-DETR.
- 🎯 Both models generalized well on dense retail shelf datasets.
- 🔄 The Streamlit application allows switching between both detection models.

---

## 🧠 Models Used

| Model | Purpose |
|--------|---------|
| YOLOv8 | Product Detection |
| RT-DETR | Dense Product Detection |
| ResNet50 | Product Classification |

---

## 📂 Project Structure

```text
retail-shelf-detection-system
│
├── app.py
├── utils.py
├── requirements.txt
│
├── models/
│
├── notebooks/
│   ├── train_yolo.ipynb
│   ├── train_rtdetr.ipynb
│   └── train_resnet50.ipynb
│
├── results/
│   ├── YOLO/
│   ├── RTDETR/
│   └── screenshots/
│
└── README.md
```

---

## 🚀 Installation

```bash
git clone https://github.com/AmmarAlrousan/retail-shelf-detection-system.git

cd retail-shelf-detection-system

pip install -r requirements.txt
```

---

## ▶️ Run

```bash
streamlit run app.py
```

---

## 🏋️ Training

Training notebooks are available in:

```text
notebooks/
```

including:

- YOLOv8 training
- RT-DETR training
- ResNet50 training

---

## 📈 Results

The repository includes evaluation outputs for both object detection models:

- Precision–Recall curves
- Precision curves
- Recall curves
- F1-score curves
- Confusion matrices
- Validation prediction samples
- Training logs

All evaluation files are available in:

```text
results/
```

---

## 🛠 Technologies

- Python
- PyTorch
- Ultralytics
- RT-DETR
- YOLOv8
- ResNet50
- OpenCV
- Streamlit

---

## 🔮 Future Improvements

- Video stream support
- Multi-camera deployment
- Automatic inventory estimation
- Shelf compliance monitoring
- Product counting
- Edge-device optimization

---

## 📄 License

MIT License
