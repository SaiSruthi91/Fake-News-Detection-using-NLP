# 📰 Fake News Detection using NLP 🔍

**End-to-End Fake News Classification with Machine Learning and Flask**

![License](https://img.shields.io/badge/license-MIT-green.svg)  
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)  
![NLP](https://img.shields.io/badge/NLP-Scikit--learn%20%7C%20TFIDF-orange.svg)  
![Status](https://img.shields.io/badge/status-Completed-brightgreen.svg)

---

## 🧠 Project Overview

This project implements a **Fake News Detection System** using Natural Language Processing (NLP) techniques. It leverages **TF-IDF vectorization** to convert news text into meaningful numerical features and classifies it as **Fake** or **Real** using a machine learning model.

The application uses a **Flask backend** for processing and a minimal **HTML/CSS frontend** for user interaction.

---

## ⚙️ Technologies Used

### 🧪 Backend

- Python 3.8+
- Flask (Web Framework)
- Scikit-learn (ML Model & Preprocessing)
- Pandas & NumPy
- Joblib (Model Serialization)

### 📝 NLP Techniques

- Lowercasing & punctuation removal
- Stopword removal
- TF-IDF Vectorization

### 🌐 Frontend

- HTML5
- CSS3 (in `static/style.css`)
- Jinja2 (Template Engine)

---

## 📁 Project Structure

```

Fake-News-Detection-using-NLP/
├── app/
│   ├── **init**.py          # App initialization
│   ├── model.py             # Load pre-trained model
│   ├── preprocess.py        # Clean text and vectorize
│   └── routes.py            # Flask route handlers
│
├── static/
│   └── style.css            # CSS for UI styling
│
├── templates/
│   └── index.html           # Frontend page
│
├── train\_model.py           # Training script
├── app.py                   # Application entry point
├── requirements.txt         # Dependency list
├── README.md                # Project documentation
└── .gitignore

```

---

## 📊 Dataset

The dataset used for this project was obtained from Kaggle:

🔗 [Fake News Detection Dataset – by Bhavik Jikadara](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection)

### Dataset Details:
- Format: CSV
- Columns: `title`, `text`, `label`
  - `label` = 0 (Fake), 1 (Real)
- Ideal for binary text classification using NLP

#### How to use:

1. Download the dataset from the link above.
2. Create a `data/` folder in your project directory.
3. Place the dataset inside:
```

Fake-News-Detection-using-NLP/
└── data/
└── fake\_news\_dataset.csv

````
4. Modify your training script (e.g., `train_model.py`) to load it:
```python
df = pd.read_csv("data/fake_news_dataset.csv")
````

---

## 🖼️ Screenshots
| Output                                 |
![WhatsApp Image 2025-07-01 at 19 32 43_8df5419d](https://github.com/user-attachments/assets/049ddbf6-dea9-45a5-bbf2-8aa727bb82b7)
| Input News Text 1                      | 
| India got independence in 1947         | 
| Prediction Output 1                    |
![WhatsApp Image 2025-07-01 at 19 33 13_6dc1aa84](https://github.com/user-attachments/assets/6953128e-49af-42b1-a035-b5efd827c76f)|
| -------------------------------------- |
| Input News Text 2                      | 
| 5G Towers Are Spreading COVID-19       | 
| Prediction Output 2                    |
![WhatsApp Image 2025-07-01 at 19 33 42_972e0d49](https://github.com/user-attachments/assets/76c48934-1ac6-4ad2-ad88-0f586decbc0f)|
| -------------------------------------- |

## 🔍 Features

✅ Detects fake news using a trained ML model
✅ TF-IDF vectorization for text feature extraction
✅ Clean user interface with Flask and HTML
✅ Lightweight and modular design
✅ Easily extendable to other classifiers or vectorizers

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SaiSruthi91/Fake-News-Detection-using-NLP.git
cd Fake-News-Detection-using-NLP
```

### 2️⃣ Set Up Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Mac/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Start the Flask App

```bash
python app.py
```

Access the app at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 Model Training (Optional)

If you'd like to retrain the model on the same or updated dataset:

```bash
python train_model.py
```

This script will:

* Load and clean data
* Perform TF-IDF vectorization
* Train a classification model
* Save the model using `joblib`

---

## 🛠️ Future Enhancements

* Use deep learning models (e.g., LSTM, BERT)
* Add support for multilingual datasets
* Upload `.docx`, `.pdf`, or web links
* Visual analytics of fake vs. real articles

---

## 📬 Contact

**Sai Sruthi Karnatakapu**
📧 [k.saisruthi913@gmail.com](mailto:k.saisruthi913@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/saisruthi-karnatakapu/)

---
