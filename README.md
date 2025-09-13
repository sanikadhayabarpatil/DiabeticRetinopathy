# 🩺 Diabetic Retinopathy Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)  
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview
This project applies **Convolutional Neural Networks (CNNs)** to classify **retinal fundus images** for **Diabetic Retinopathy (DR)** detection.  

Diabetic Retinopathy is a leading cause of blindness, and **early detection** can prevent vision loss.  
Using **EfficientNetB3 (transfer learning)**, the model achieves **~85% accuracy** on **3,000+ retinal images**, supporting reliable automated screening for DR.

---

## 🚀 Features
- ✅ Deep learning classifier using **EfficientNetB3**  
- ✅ Trained on **3,000+ retinal fundus images**  
- ✅ Achieved **~85% accuracy** on validation/test sets  
- ✅ **Gaussian-filtered dataset preprocessing**  
- ✅ Regularization with **Dropout + L2/L1 penalties**  
- ✅ Evaluation via **confusion matrix & classification report**  

---

## 🗂 Dataset
- **Source**: [Kaggle Diabetic Retinopathy (224x224 Gaussian Filtered)](https://www.kaggle.com/datasets/paultimothymooney/diabetic-retinopathy-resized)  
- **Classes**:  
  - 0: No DR  
  - 1: Mild  
  - 2: Moderate  
  - 3: Severe  
  - 4: Proliferative DR  

- **Preprocessing**:  
  - Images resized to **224×224 RGB**  
  - Gaussian filtered version used for consistency  

---

## 🏗 Model Architecture
- **Base Model**: `EfficientNetB3` (ImageNet pretrained)  
- **Custom Layers**:
  - Batch Normalization  
  - Dense (256 units, ReLU, L2/L1 regularization)  
  - Dropout (0.45)  
  - Output: Dense Softmax (multi-class)  

- **Training Setup**:  
  - Optimizer: `Adamax (lr=0.001)`  
  - Loss: `Categorical Crossentropy`  
  - Metrics: `Accuracy`  
  - Epochs: up to 40 with adaptive LR + early stopping  

---

## 📊 Results
- **Accuracy**: ~85% on validation/test sets  
- **Performance Metrics**:
  - ✅ Confusion Matrix  
  - ✅ Classification Report (Precision, Recall, F1)  
  - ✅ Accuracy/Loss Curves  

<p align="center">
  <img src="https://img.icons8.com/color/96/graph.png" width="80" /><br/>
  <i>Training & Evaluation Visualizations</i>
</p>

---

## ⚙️ Installation & Setup
Clone the repository:
```bash
git clone https://github.com/<your-username>/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
pip install -r requirements.txt
jupyter notebook diabetic-retinopathy-detection-using-deep-learning.ipynb
```
## 📚 References

- Kaggle Diabetic Retinopathy Dataset

- EfficientNet: Tan & Le, 2019

- TensorFlow / Keras Docs

## 👨‍💻 Author

Sanika Dhayabar Patil 
