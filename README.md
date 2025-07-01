# 📰 Fake News Detection using NLP 🔍

**End-to-End Fake News Classification with Machine Learning and Flask**

![License](https://img.shields.io/badge/license-MIT-green.svg)  
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)  
![NLP](https://img.shields.io/badge/NLP-Scikit--learn%20%7C%20TFIDF-orange.svg)  
![Status](https://img.shields.io/badge/status-Completed-brightgreen.svg)

---

## 🧠 Project Overview

This project implements a **Fake News Detection System** using Natural Language Processing (NLP) techniques. It uses **TF-IDF vectorization** to convert text into numerical features and a supervised ML model to classify news articles as **Fake** or **Real**.

The project follows a modular design with a Flask-based backend and an HTML interface for user input and output rendering.

---

## ⚙️ Technologies Used

### 🧪 Backend

- Python 3.8+
- Flask (Web framework)
- Scikit-learn (ML Model & Preprocessing)
- Pandas & NumPy
- Joblib (Model Serialization)

### 📝 NLP Techniques

- Text Cleaning (Lowercasing, punctuation removal)
- Tokenization & Stopword Removal
- TF-IDF Vectorization

### 🌐 Frontend

- HTML5
- CSS3 (Stored in `/static/style.css`)
- Jinja2 Template Rendering

---

## 📁 Project Structure

```

Fake-News-Detection-using-NLP/
├── app/
│   ├── **init**.py          # App initialization
│   ├── model.py             # Model loading logic
│   ├── preprocess.py        # Text cleaning & TF-IDF logic
│   └── routes.py            # Flask routes
│
├── static/
│   └── style.css            # Stylesheet for UI
│
├── templates/
│   └── index.html           # HTML template
│
├── train\_model.py           # ML model training script
├── app.py                   # App entry point
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .gitignore

````

---

## 🖼️ Screenshots

| Input Interface              | Prediction Output          |
|-----------------------------|----------------------------|
| ![Upload](./screenshots/input_ui.png) | ![Result](./screenshots/output_result.png) |

> 📌 Place your screenshots in a `screenshots/` folder and update paths above.

---

## 🔍 Features

✅ Classifies input news text as **Fake** or **Real**  
✅ Clean and minimal web interface using HTML & CSS  
✅ Modular Flask backend  
✅ Pre-trained model loaded using `joblib`  
✅ TF-IDF based feature extraction  
✅ Separate training pipeline (`train_model.py`)

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SaiSruthi91/Fake-News-Detection-using-NLP.git
cd Fake-News-Detection-using-NLP
````

### 2️⃣ Set Up Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Linux/Mac

pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## 🧠 Model Training (Optional)

If you wish to retrain the model:

```bash
python train_model.py
```

This script will:

* Load the dataset
* Clean and vectorize text
* Train a classifier (e.g., Logistic Regression or PassiveAggressiveClassifier)
* Save model and vectorizer to disk

---

## 📬 Contact

**Sai Sruthi Karnatakapu**
📧 [k.saisruthi913@gmail.com](mailto:k.saisruthi913@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/saisruthikarnatakapu/)

---

## 🏷️ Tags

**Fake News Detection | NLP | Flask App | TF-IDF | ML Model | Text Classification**

