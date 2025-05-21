# EOG-Based Calculator 👁️🧮

This project implements a calculator that can be controlled using **EOG (Electrooculography) signals**. It’s designed to assist individuals with motor impairments by enabling calculations through eye movements.

---

### 🧠 Project Overview

We built a machine learning pipeline to classify EOG-based eye gestures and used these gestures to control a **GUI calculator**. The interface is built using **Tkinter**, and we used **Streamlit** for deployment.

---

### 🔍 Key Features

- ✅ Eye gesture recognition using EOG signals from `.txt` files
- ✅ ML model trained to classify eye movements (e.g., left, right, blink)
- ✅ Saved model in `.pkl` format for reuse
- ✅ GUI built with **Tkinter**
- ✅ Deployment and automation with **Streamlit**

---

### 🧪 AI & Model Training

Implemented in `EOG2.ipynb`:

- **Input Format:** EOG signals loaded from `.txt` files
- **Preprocessing:** Cleaning and preparing EOG data
- **Feature Extraction:** Statistical features from eye movements
- **Model Training:** Supervised classification (e.g., Random Forest / SVM)
- **Evaluation:** Accuracy metrics and confusion matrix
- **Model Saving:** Exported using `joblib

> 📁 `model.pkl` – the trained classifier used in deployment

---

### 🖥️ GUI & Deployment

#### 🎨 GUI with Tkinter

A calculator interface built with Python’s `tkinter`:

- Visual buttons and display
- Gesture-based navigation and selection
- Integrated with the trained model for real-time interaction

#### 🚀 Deployment Using Streamlit

used to automate deployment and simulate user input:

- Load model and predict gestures
- Automate calculator interaction based on predictions
- Test automation and functional verification

---
