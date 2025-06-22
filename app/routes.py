from flask import Blueprint, render_template, request
import pickle
from .preprocess import clean_text
import os

main = Blueprint('main', __name__)

base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, '../model/fake_news_model.pkl')
vectorizer_path = os.path.join(base_dir, '../model/vectorizer.pkl')

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))


@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        
        # Preprocess
        cleaned = clean_text(news_text)
        vectorized = vectorizer.transform([cleaned])
        
        # Predict & confidence
        proba = model.predict_proba(vectorized)[0]
        confidence = round(max(proba) * 100, 2)
        label = model.predict(vectorized)[0]
        
        result = "Real News" if label == 1 else "Fake News"
        
        return render_template('result.html', prediction=result, confidence=confidence)
    
        if confidence < 60:
            result = "Uncertain — Please verify manually"

