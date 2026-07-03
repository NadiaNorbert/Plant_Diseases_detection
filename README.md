# 🌿 Plant Disease Detection using Deep Learning

An AI-powered web application that detects plant diseases from leaf images using **PyTorch** and **MobileNetV2**. The application predicts the disease, displays the confidence score, and provides precautionary measures along with weather-based recommendations.

---

## 🚀 Features

- 🌱 Detects plant diseases from uploaded leaf images
- 🤖 Deep Learning model using MobileNetV2
- 📊 Displays prediction confidence
- 🌤 Weather-based farming precautions
- ⚡ FastAPI backend
- 🖼 Image preprocessing using Pillow and Torchvision

---

## 🛠 Tech Stack

### Programming Language
- Python

### AI / Machine Learning
- PyTorch
- Torchvision
- MobileNetV2 (Transfer Learning)

### Backend
- FastAPI
- Uvicorn

### Libraries
- Pillow (PIL)
- NumPy
- Pandas
- Matplotlib

### Platform
- Kaggle (Model Training)
- VS Code

---

## 📂 Project Structure

```
Plant_Diseases_detection/
│
├── app.py
├── weather.py
├── requirements.txt
├── README.md
├── .gitignore
├── model/
│   └── plant_disease_model.pth
├── templates/
├── static/
└── images/
```

---

## 📷 Workflow

1. Upload a plant leaf image
2. Image preprocessing
3. Deep Learning model prediction
4. Disease detection
5. Confidence score generation
6. Display precautions and weather recommendations

---

## 🧠 AI Concepts Used

- Deep Learning
- Convolutional Neural Networks (CNN)
- Transfer Learning
- Image Classification
- Model Inference

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/NadiaNorbert/Plant_Diseases_detection.git
```

Move into the project

```bash
cd Plant_Diseases_detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
uvicorn app:app --reload
```

---

## 📊 Future Enhancements

- Disease severity prediction
- Multi-language support
- Mobile application
- Prediction history
- Dashboard and analytics

---
