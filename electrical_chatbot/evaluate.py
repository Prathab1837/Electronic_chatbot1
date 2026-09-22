
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


with open("dataset.json", "r") as file:
    data = json.load(file)


patterns = []
tags = []


for intent in data["intents"]:

    for pattern in intent["patterns"]:

        patterns.append(pattern)
        tags.append(intent["tag"])


X_train, X_test, y_train, y_test = train_test_split(
    patterns,
    tags,
    test_size=0.2,
    random_state=42,
    stratify=tags
)


vectorizer = TfidfVectorizer()


X_train_vectorized = vectorizer.fit_transform(
    X_train
)

X_test_vectorized = vectorizer.transform(
    X_test
)


model = LogisticRegression()

model.fit(
    X_train_vectorized,
    y_train
)


predictions = model.predict(
    X_test_vectorized
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("Accuracy:", accuracy)


print("\nClassification Report:\n")


print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


print("\nWrong Predictions:\n")


wrong_count = 0


for question, actual, predicted in zip(
    X_test,
    y_test,
    predictions
):

    if actual != predicted:

        wrong_count += 1

        print("Question :", question)
        print("Actual   :", actual)
        print("Predicted:", predicted)

        print("-" * 50)


print(
    "\nTotal wrong predictions:",
    wrong_count
)

