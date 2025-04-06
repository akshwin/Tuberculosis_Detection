

# TBXPro: Tuberculosis Detection using CNN-BiLSTM Hybrid Model

This repository presents **TBXPro**, a deep learning model designed to detect **Tuberculosis (TB)** from chest X-ray images. TBXPro combines the spatial feature extraction power of **Convolutional Neural Networks (CNNs)** with the temporal pattern recognition of **Bidirectional LSTM (Bi-LSTM)** to enhance diagnostic accuracy.

---

## 🔬 Project Overview

Tuberculosis is a life-threatening bacterial infection that primarily affects the lungs. Early diagnosis is crucial to control its spread and begin treatment. TBXPro aims to assist healthcare professionals by providing a reliable AI-based system that can analyze chest X-rays and predict TB infection with high accuracy.

---

## 🛠️ Key Features

- 📁 Preprocessing pipeline for chest X-ray enhancement and augmentation
- 🧠 **TBXPro**: A hybrid model combining **CNN** and **Bi-LSTM**
- 📊 Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- 
---

## 📂 Dataset

We use a combination of open-source chest X-ray datasets for training and evaluation:

- **Montgomery County X-ray Set** (USA)
- **Shenzhen Hospital X-ray Set** (China)
- (Optional) **NIH Chest X-ray14** dataset

All X-rays are labeled as:
- **TB Positive**
- **Normal (TB Negative)**

---

## 🧠 Model Architecture - TBXPro

**TBXPro** is a hybrid architecture:

- **CNN layers** extract spatial and structural features from X-ray images.
- **Bi-LSTM layers** capture long-range dependencies and subtle variations in features.
- Fully connected layers perform final classification.

This fusion boosts the model’s ability to handle complex patterns in medical imaging.

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TBXPro-TB-Detection.git
   cd TBXPro-TB-Detection
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run training:
   ```bash
   python train_tb_model.py
   ```

4. Evaluate and visualize results:
   ```bash
   python evaluate.py
   ``


## 🔮 Future Enhancements

- 🖥️ Deploy TBXPro as a web/mobile diagnostic tool
- 🧪 Add support for multi-disease detection (e.g., Pneumonia, COVID-19)
- 🧠 Explore attention mechanisms or transformers for further accuracy gains

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 📧 Contact

- **Name:** Akshwin T  
- **Email:** your_email@example.com  
- **LinkedIn:** [Your LinkedIn Profile](https://www.linkedin.com/)
