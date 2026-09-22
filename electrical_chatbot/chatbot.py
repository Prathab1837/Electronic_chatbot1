import json
import joblib
import streamlit as st
from pathlib import Path

# Get the folder containing this Python file

BASE_DIR = Path(__file__).resolve().parent

# Load dataset

with open(BASE_DIR / "dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Load trained model

model = joblib.load(BASE_DIR / "model.pkl")

vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")

# Streamlit page configuration

st.set_page_config(
page_title="Electrical Learning Chatbot",
page_icon="⚡"
)

# Application title

st.title("⚡ Electrical Learning Chatbot")

st.write(
"Ask questions about electrical engineering topics."
)

# Create chat history

if "messages" not in st.session_state:

```
    st.session_state.messages = []
```

# Display previous messages

for message in st.session_state.messages:

```
    with st.chat_message(message["role"]):

        st.write(message["content"])
```

# User input

question = st.chat_input(
"Ask an electrical question..."
)

if question:

```
# Store user message

    st.session_state.messages.append(
    {
        "role": "user",
        "content": question
    }
)


# Display user message

with st.chat_message("user"):

    st.write(question)


# Convert question to TF-IDF

question_vector = vectorizer.transform(
    [question]
)


# Calculate probabilities

probabilities = model.predict_proba(
    question_vector
)

highest_probability = probabilities.max()


# Predict intent

prediction = model.predict(
    question_vector
)

predicted_tag = prediction[0]


# Unknown / low-confidence handling

if highest_probability < 0.30:

    response = (
        "I don't know the answer to that yet. "
        "I currently focus on electrical engineering topics."
    )

else:

    response = (
        "I don't know the answer to that yet."
    )


    # Find response for predicted intent

    for intent in data["intents"]:

        if intent["tag"] == predicted_tag:

            response = intent["responses"][0]

            break


# Store bot response

st.session_state.messages.append(
    {
        "role": "assistant",
        "content": response
    }
)


# Display bot response

with st.chat_message("assistant"):

    st.write(response)
```
