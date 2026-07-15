import joblib
from preprocessing import clean_text

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# User input
news = input("Enter News: ")

# Clean the text
news = clean_text(news)

# Convert text into TF-IDF vector
news_vector = vectorizer.transform([news])

# Predict
prediction = model.predict(news_vector)

# Display result
if prediction[0] == 1:
    print("\nPrediction: REAL NEWS")
else:
    print("\nPrediction: FAKE NEWS")