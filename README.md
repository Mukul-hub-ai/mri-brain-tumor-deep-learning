# mri-brain-tumor-deep-learning
Deep learning-based Brain Tumor MRI Classification system built using PyTorch and ResNet50. The model classifies MRI images into glioma, meningioma, pituitary, and no tumor categories. Includes data preprocessing, transfer learning, model evaluation, and a Streamlit web app for real-time predictions with confidence scores.
# 🧠 Brain Tumor MRI Image Classification

A deep learning-based medical imaging project for classifying brain MRI images into multiple tumor categories using Transfer Learning (ResNet50) and deployed via Streamlit.

---

## 📌 Project Overview

This project builds an AI-powered classification system capable of detecting and categorizing brain tumors from MRI images. The model uses a pretrained ResNet50 architecture fine-tuned for medical image classification.

The application allows users to upload MRI scans and receive real-time tumor predictions with confidence scores.

---

## 🎯 Problem Statement

To develop a deep learning solution that accurately classifies brain MRI images into:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The system aims to assist in AI-assisted medical diagnosis and improve early detection workflows.

---

## 🛠 Technologies Used

- Python
- PyTorch
- Torchvision
- Transfer Learning (ResNet50)
- NumPy
- PIL
- Streamlit
- Deep Learning
- Medical Image Processing

---

## 📂 Project Structure

├── main_project.ipynb # Model training & evaluation
├── app.py # Streamlit web application
├── best_resnet50.pth # Trained model weights
├── README.md # Project documentation


---

## 🔬 Model Architecture

- Base Model: ResNet50 (Pretrained on ImageNet)
- Final Layer Modified for 4-class classification
- Softmax activation for probability output
- Trained using CrossEntropyLoss

---

## 📊 Dataset

Brain MRI Tumor Dataset containing categorized MRI images:
- Glioma
- Meningioma
- Pituitary
- No Tumor

Images resized to 224x224 and normalized before training.

---

## 📈 Model Evaluation

Model performance evaluated using:

- Accuracy
- Loss Curves
- Class Probabilities
- Softmax Confidence Score

---

## 🌐 Streamlit Web Application

The project includes a user-friendly web interface built with Streamlit.

### Features:
- Upload MRI image (jpg/png/jpeg)
- Real-time tumor classification
- Confidence percentage display
- Class probability breakdown

### Run the App

```bash
streamlit run app.py

👨‍💻 Author
Mukul
Deep Learning & AI Enthusiast


📌 Disclaimer

This project is for educational and research purposes only. It is not intended to replace professional medical diagnosis.
