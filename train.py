import json
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    Dense, Dropout, GlobalMaxPooling1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load dataset
# -----------------------------
with open("intents.json") as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        training_sentences.append(pattern)
        training_labels.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

num_classes = len(labels)

# -----------------------------
# Encode labels
# -----------------------------
label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(training_labels)

# -----------------------------
# Tokenization
# -----------------------------
vocab_size = 2000
embedding_dim = 128
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)

sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(
    sequences,
    maxlen=max_len,
    padding="post",
    truncating="post"
)

# -----------------------------
# Bi-LSTM + ReLU Model
# -----------------------------
model = Sequential([
    Input(shape=(max_len,)),

    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True
    ),

    Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    ),

    GlobalMaxPooling1D(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(num_classes, activation="softmax")
])

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------
early_stop = EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    padded_sequences,
    np.array(training_labels),
    epochs=50,
    batch_size=8,
    callbacks=[early_stop]
)

# -----------------------------
# Save model and preprocessors
# -----------------------------
model.save("chat_model.keras")

with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("label_encoder.pickle", "wb") as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Training complete. Model saved successfully.")
