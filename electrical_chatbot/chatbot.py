import json
import joblib
import streamlit as st
from pathlib import Path

# --------------------------------------------------

# Get the folder containing this Python file

# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------

# Load dataset

# --------------------------------------------------

dataset_path = BASE_DIR / "dataset.json"

with open(dataset_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# --------------------------------------------------

# Load trained model and TF-IDF vectorizer

# --------------------------------------------------

model_path = BASE_DIR / "model.pkl"
vectorizer_path = BASE_DIR / "vectorizer.pkl"

model = joblib.load(model_path)

vectorizer = joblib.load(vectorizer_path)

# --------------------------------------------------

# Streamlit configuration

# --------------------------------------------------

st.set_page_config(
page_title="Electrical Learning Chatbot",
page_icon="⚡"
)

# --------------------------------------------------

# Title

# --------------------------------------------------

st.title("⚡ Electrical Learning Chatbot")

st.write(
"Ask questions about electrical engineering topics."
)

# --------------------------------------------------

# Create chat history

# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------

# Display previous messages

# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --------------------------------------------------

# Get user question

# --------------------------------------------------

user_input = st.chat_input(
"Ask an electrical question..."
)

# --------------------------------------------------

# Process question

# --------------------------------------------------

if user_input is not None:
# Convert the input explicitly to a normal Python string
    question = str(user_input).strip()


# Ignore empty input

if question is Null:
    st.warning("Please enter a question.")

    st.stop()


# --------------------------------------------------
# Store user message
# --------------------------------------------------

st.session_state.messages.append(
    {
        "role": "user",
        "content": question
    }
)


# --------------------------------------------------
# Display user message
# --------------------------------------------------

with st.chat_message("user"):

    st.write(question)


# --------------------------------------------------
# Convert question into TF-IDF vector
# --------------------------------------------------

try:

    question_vector = vectorizer.transform(
        [question]
    )

except Exception as error:

    st.error(
        "An error occurred while converting your question "
        "into a TF-IDF vector."
    )

    st.code(
        str(error)
    )

    st.stop()


# --------------------------------------------------
# Predict intent
# --------------------------------------------------

probabilities = model.predict_proba(
    question_vector
)

highest_probability = probabilities.max()


prediction = model.predict(
    question_vector
)

predicted_tag = prediction[0]


# --------------------------------------------------
# Find response
# --------------------------------------------------

if highest_probability < 0.30:

    response = (
        "I don't know the answer to that yet. "
        "I currently focus on electrical engineering topics."
    )

else:

    response = (
        "I don't know the answer to that yet."
    )

    for intent in data["intents"]:

        if intent["tag"] == predicted_tag:

            response = intent["responses"][0]

            break


# --------------------------------------------------
# Store bot response
# --------------------------------------------------

st.session_state.messages.append(
    {
        "role": "assistant",
        "content": response
    }
)


# --------------------------------------------------
# Display bot response
# --------------------------------------------------

with st.chat_message("assistant"):

    st.write(response)
