import json
import joblib

from utils import find_response


with open("dataset.json", "r") as file:
    data = json.load(file)


model = joblib.load("model.pkl")

vectorizer = joblib.load("vectorizer.pkl")


previous_tag = None


def get_response(question):

    global previous_tag

    question_vector = vectorizer.transform([question])

    probabilities = model.predict_proba(question_vector)

    print("\nAll intent probabilities:")

    for tag, probability in zip(model.classes_, probabilities[0]):
        print(f"{tag}: {probability:.4f}")

    highest_probability = probabilities.max()

    prediction = model.predict(question_vector)

    predicted_tag = prediction[0]

    print("Confidence:", highest_probability)
    print("Predicted tag:", predicted_tag)

    if highest_probability < 0.30:

        if previous_tag is not None:

            follow_up_words = [
                "it",
                "its",
                "this",
                "that",
                "they",
                "their",
                "them"
            ]

            words = question.lower().split()

            is_follow_up = False

            for word in follow_up_words:

                if word in words:
                    is_follow_up = True
                    break

            if is_follow_up:
                predicted_tag = previous_tag

            else:
                return "I don't know the answer to that yet."

        else:
            return "I don't know the answer to that yet."

    previous_tag = predicted_tag

    return find_response(data, predicted_tag)


print("Electrical Chatbot")
print("Type 'quit' to exit.")


while True:

    question = input("\nYou: ")

    if question.lower() == "quit":
        print("Bot: Goodbye!")
        break

    response = get_response(question)

    print("Bot:", response)
