import json
import pickle
import random
import numpy as np
from tensorflow import keras
from colorama import Fore, Style, init

init(autoreset=True)

# ----------------------------
# Load resources once
# ----------------------------
def load_resources():
    with open("intents.json") as file:
        intents = json.load(file)

    model = keras.models.load_model("chat_model.keras")

    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pickle", "rb") as f:
        label_encoder = pickle.load(f)

    # Convert intents list → dictionary for fast lookup
    intent_responses = {
        intent["tag"]: intent["responses"]
        for intent in intents["intents"]
    }

    return model, tokenizer, label_encoder, intent_responses


# ----------------------------
# Predict intent
# ----------------------------
def predict_intent(text, model, tokenizer, label_encoder, max_len=20):
    sequence = tokenizer.texts_to_sequences([text])
    padded = keras.preprocessing.sequence.pad_sequences(
        sequence, truncating="post", maxlen=max_len
    )

    predictions = model.predict(padded, verbose=0)
    tag_index = np.argmax(predictions)
    tag = label_encoder.inverse_transform([tag_index])[0]

    confidence = predictions[0][tag_index]
    return tag, confidence


# ----------------------------
# Chat loop
# ----------------------------
def chat():
    model, tokenizer, label_encoder, intent_responses = load_resources()

    print(Fore.YELLOW + "Start messaging with the bot (type 'quit' to stop)")

    while True:
        try:
            user_input = input(Fore.LIGHTBLUE_EX + "User: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print(Fore.YELLOW + "Goodbye!")
                break

            tag, confidence = predict_intent(
                user_input, model, tokenizer, label_encoder
            )

            # Optional confidence threshold
            if confidence < 0.5:
                response = "I'm not sure I understood that. Can you rephrase?"
            else:
                response = random.choice(intent_responses.get(tag, ["I don't understand."]))

            print(Fore.GREEN + "ChatBot:", response)

        except KeyboardInterrupt:
            print("\n" + Fore.YELLOW + "Chat ended by user.")
            break


# ----------------------------
# Run chatbot
# ----------------------------
if __name__ == "__main__":
    chat()
