import pickle
import os

# Load model and vectorizer
model_path = os.path.join('model', 'fake_news_model.pkl')
vectorizer_path = os.path.join('model', 'vectorizer.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Function to predict news
def predict_news(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return 'Fake News' if prediction[0] == 0 else 'Real News'
