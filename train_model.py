import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import os
import gc  # Import garbage collector

# Load CSV file paths
fake_path = os.path.join('data', 'Fake_news.csv')
true_path = os.path.join('data', 'True_news.csv')

# Load datasets with explicit encoding
print("🔄 Loading datasets...")
fake_df = pd.read_csv(fake_path, on_bad_lines='skip', engine='python', encoding='latin1')
true_df = pd.read_csv(true_path, on_bad_lines='skip', engine='python', encoding='latin1')

# Assign labels
fake_df['label'] = 0  # Fake news
true_df['label'] = 1  # Real news

# Combine and shuffle
print("🔄 Combining datasets...")
df = pd.concat([fake_df, true_df], axis=0).sample(frac=1).reset_index(drop=True)
df.columns = df.columns.str.strip()

# Clean up memory
del fake_df, true_df
gc.collect()

# Show label distribution
print("\n🧾 Label distribution:\n", df['label'].value_counts())

# Drop empty or NaN texts
initial_count = len(df)
df = df.dropna(subset=['text'])
df = df[df['text'].str.strip() != '']
print(f"🧹 Removed {initial_count - len(df)} empty rows. New size: {len(df)}")

# Features and labels
X = df['text']
y = df['label']

# Clean up more memory
del df
gc.collect()

# Use HashingVectorizer instead of TF-IDF to save memory
print("\n🔧 Vectorizing text data...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.95,
    min_df=5,  # Increased min_df to reduce features
    ngram_range=(1, 1),  # Use only unigrams
    max_features=15000   # Limit number of features
)
X_vectorized = vectorizer.fit_transform(X)

# Stratified split
print("\n✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

# Clean up memory
del X_vectorized, X, y
gc.collect()

# Apply SMOTE with lower sampling strategy
print("\n🔁 Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42, sampling_strategy=0.75)  # Don't create perfect balance
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Clean up memory
del X_train, y_train
gc.collect()

# Train Naive Bayes
print("\n🤖 Training model...")
model = MultinomialNB(alpha=0.5)  # Simplified parameters
model.fit(X_train_resampled, y_train_resampled)

# Evaluation
print("\n📊 Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
os.makedirs('model', exist_ok=True)
pickle.dump(model, open('model/fake_news_model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))

print("\n💾 Model and vectorizer saved successfully!")