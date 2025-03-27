import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# 1. Carregar dados
df = pd.read_csv("data/Dataset_AI_vs_Human.csv", sep=",")
df["Label"] = df["Label"].astype(str).str.strip().str.upper()
df = df[df["Label"].isin(["AI", "HUMAN"])]

# 2. Preparar dados
texts = df["Text"].astype(str).tolist()
labels = df["Label"].tolist()

# Codificar os rótulos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)

# 3. Tokenização
max_words = 10000
max_len = 100

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# 4. Criar modelo com Embeddings
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Treinar
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

# 6. Avaliar
y_pred_probs = model.predict(X_test_pad)
y_pred = (y_pred_probs > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 7. Guardar modelo e tokenizer
model.save("best_model_embedding.keras")
with open("tokenizer_embedding.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Modelo e tokenizer guardados com sucesso.")
