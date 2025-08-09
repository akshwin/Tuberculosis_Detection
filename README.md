
<h1 align="center">🫁 TB-EnsembleX: Ensemble-Based Transfer Learning for Tuberculosis Detection</h1>

<p align="center">
  <a href="https://tuberculosis-detector.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀 Launch%20App-Try%20Now-brightgreen?style=for-the-badge" alt="Streamlit App" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-Ensemble%20CNN%20+%20Voting%20Classifier-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Accuracy-99%25-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Technique-Transfer%20Learning%20+%20PCA%20+%20SMOTE-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

> **TB-EnsembleX** is a high-performance, AI-driven ensemble architecture designed for **automated tuberculosis detection** from chest X-ray images. It combines multiple deep CNN feature extractors, dimensionality reduction, and a voting-based classifier to provide accurate and robust TB classification.

---

## 📌 Table of Contents

- [📖 Abstract](#-abstract)
- [🚀 Key Features](#-key-features)
- [📂 Dataset Used](#-dataset-used)
- [🧠 Methodology](#-methodology)
- [📈 Performance Metrics](#-performance-metrics)
- [⚙️ Installation & Usage](#️-installation--usage)
- [📊 Evaluation & Results](#-evaluation--results)
- [🛣 Future Scope](#-future-scope)
- [📜 License](#-license)
- [📧 Contact](#-contact)

---

## 📖 Abstract

This study introduces **TB-EnsembleX**, a novel ensemble-based transfer learning architecture for automated **tuberculosis detection** from chest X-ray images. The architecture integrates multiple pretrained CNNs including **VGG16**, **VGG19**, **InceptionV3**, and **Xception** to extract diverse and rich features. These features are concatenated into a unified feature vector.

To address class imbalance, **SMOTE (Synthetic Minority Oversampling Technique)** is applied during preprocessing. Then, **PCA (Principal Component Analysis)** is used to reduce feature dimensionality while retaining essential variance. A **Voting-based ensemble classifier** consisting of multiple Logistic Regression models is trained on the PCA-transformed space.

The model achieves an **impressive 99% accuracy**, showcasing its robustness and potential in **computer-aided TB screening**.

---

## 🚀 Key Features

✅ Ensemble of four powerful pretrained CNNs (VGG16, VGG19, InceptionV3, Xception)  
✅ Transfer learning for improved feature generalization  
✅ SMOTE for class imbalance handling  
✅ PCA for dimensionality reduction  
✅ Voting classifier for reliable final predictions  
✅ Evaluation with comprehensive metrics and confusion matrix

---

## 📂 Dataset Used

We use combined chest X-ray datasets for binary classification:

- ✅ **TB Positive**
- ❌ **TB Negative (Normal)**

Sources:
- 🏥 Montgomery County X-ray Set
- 🏥 Shenzhen Hospital X-ray Set

All images are resized to 224×224, augmented, normalized, and preprocessed for consistent input to the CNN models.

---

## 🧠 Methodology

**TB-EnsembleX** follows a multi-stage pipeline:

```

  Input Image (224x224x3)
          ↓
  Feature Extraction using:
      * VGG16
      * VGG19
      * InceptionV3
      * Xception
          ↓
  Concatenation of Feature Vectors
          ↓
  SMOTE (Synthetic Minority Oversampling)
          ↓
  PCA (Dimensionality Reduction)
          ↓
  Voting Classifier (Logistic Regression)
          ↓  
  TB / No TB Prediction

````

---

## 📈 Performance Metrics

| Metric       | Score       |
|--------------|-------------|
| Accuracy     | **99.0%**   |
| Precision    | **98.8%**   |
| Recall       | **99.2%**   |
| F1-Score     | **99.0%**   |
| AUC-ROC      | **0.995**   |

> 🧪 These metrics confirm the model's outstanding diagnostic ability and generalizability.

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/TB-EnsembleX.git
cd TB-EnsembleX
````

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Training Pipeline

```bash
python train_model.py
```

---

## 📊 Evaluation & Results

The results include:

* ✅ Confusion Matrix
* 📉 ROC & AUC Curves
* 🧮 Classification Report
* 📈 Visualizations for PCA + Classifier Decisions

---

## 🛣 Future Scope

* 💡 Incorporate additional classifiers like SVM, XGBoost for fusion
* 🔍 Add explainability via Grad-CAM
* 🌍 Web/Mobile deployment using Streamlit or TensorFlow Lite
* 🦠 Multi-label support for TB, Pneumonia, COVID-19, etc.

---

## 📜 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for more details.

---

## 📧 Contact

> 👨‍💻 **Developed by Akshwin T**
> 📧 [akshwin.projects@gmail.com](mailto:akshwin.projects@gmail.com)
> 🌐 [LinkedIn](https://www.linkedin.com/in/akshwin/)

---

## ⭐ Star the Repository

If you found this project helpful, please ⭐ star the repository to support further research and development.

---
