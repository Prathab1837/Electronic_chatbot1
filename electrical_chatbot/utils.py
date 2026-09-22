def find_response(data, predicted_tag):

    for intent in data["intents"]:

        if intent["tag"] == predicted_tag:
            return intent["responses"][0]

    return "I don't know the answer to that yet."