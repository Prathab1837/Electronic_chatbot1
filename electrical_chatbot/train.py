
import json
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


with open("dataset.json", "r") as file:
    data = json.load(file)


patterns = []
tags = []


for intent in data["intents"]:

    for pattern in intent["patterns"]:

        patterns.append(pattern)
        tags.append(intent["tag"])


vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(patterns)


model = LogisticRegression()

model.fit(X, tags)


joblib.dump(vectorizer, "vectorizer.pkl")

joblib.dump(model, "model.pkl")


print("Training completed.")
print("Model saved as model.pkl")
print("Vectorizer saved as vectorizer.pkl")

