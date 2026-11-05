# train.py
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
fake_df = pd.read_csv("Fake.csv", on_bad_lines='skip', engine='python')
true_df = pd.read_csv("True.csv", on_bad_lines='skip', engine='python')

fake_df["label"] = 0  # Fake news
true_df["label"] = 1  # True news

df = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42).reset_index(drop=True)
print("Datasets Loaded!")

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return " ".join(text.split())

df["clean_text"] = df["text"].apply(clean_text)

# Split data
X = df["clean_text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model saved!")

# Accuracy
y_pred = model.predict(X_test_vect)
print("Accuracy:", accuracy_score(y_test, y_pred))
