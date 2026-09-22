import json
import joblib
import streamlit as st
from pathlib import Path

# --------------------------------------------------

# Get the folder containing chatbot.py

# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------

# Load dataset

# --------------------------------------------------

dataset_path = BASE_DIR / "dataset.json"

with open(dataset_path,"r",encoding="utf-8") as file:
    data = json.load(file)

# --------------------------------------------------

# Load trained model

# --------------------------------------------------

model_path = BASE_DIR / "model.pkl"

model = joblib.load(model_path)

# --------------------------------------------------

# Load TF-IDF vectorizer

# --------------------------------------------------

vectorizer_path = BASE_DIR / "vectorizer.pkl"

vectorizer = joblib.load(vectorizer_path)

# --------------------------------------------------

# Streamlit page configuration

# --------------------------------------------------

st.set_page_config(page_title="Electrical Learning Chatbot",page_icon="⚡")

# --------------------------------------------------

# Application title

# --------------------------------------------------

st.title("⚡ Electrical Learning Chatbot")

st.write("Ask questions about electrical engineering topics.")

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

# Get user input

# --------------------------------------------------

user_input = st.chat_input("Ask an electrical question...")

# --------------------------------------------------

# Process user input

# --------------------------------------------------

if user_input is not None:


# Make sure the input is a string

    if isinstance(user_input, str):
        question = str(user_input).strip()
        question = st.chat_input("Ask an electrical question...")

if question is not None and question.strip() != "":

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
# TF-IDF input
# --------------------------------------------------

tfidf_input = question


# Make sure TF-IDF receives a string

if not isinstance(tfidf_input, str):

    tfidf_input = str(tfidf_input)


# --------------------------------------------------
# Convert question to TF-IDF
# --------------------------------------------------

try:

    question_vector = vectorizer.transform(
        [tfidf_input]
    )

except Exception as error:

    st.error(
        "Error while converting the question "
        "into a TF-IDF vector."
    )

    st.write(
        "Question:",
        question
    )

    st.write(
        "Question type:",
        type(question).__name__
    )

    st.write(
        "TF-IDF input:",
        tfidf_input
    )

    st.write(
        "TF-IDF input type:",
        type(tfidf_input).__name__
    )

    st.code(
        str(error)
    )

    st.stop()


# --------------------------------------------------
# Calculate probabilities
# --------------------------------------------------

probabilities = model.predict_proba(question_vector)


# Get highest probability

highest_probability = probabilities.max()


# --------------------------------------------------
# Predict intent
# --------------------------------------------------

prediction = model.predict(question_vector)

predicted_tag = prediction[0]


# --------------------------------------------------
# Find response
# --------------------------------------------------

if highest_probability < 0.30:

    response = (
        "I don't know the answer to that yet. "
        "I currently focus on electrical "
        "engineering topics."
    )

else:

    response = (
        "I don't know the answer to that yet."
    )


    # Search for matching intent

    for intent in data["intents"]:

        if intent["tag"] == predicted_tag:

            response = intent["responses"][0]

            break


# --------------------------------------------------
# Store assistant response
# --------------------------------------------------

st.session_state.messages.append(
    {
        "role": "assistant",
        "content": response
    }
)


# --------------------------------------------------
# Display assistant response
# --------------------------------------------------

with st.chat_message("assistant"):

    st.write(response)

