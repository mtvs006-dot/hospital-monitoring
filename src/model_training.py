import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("data/patient_data.csv").dropna()

# ---------------- Logistic Regression ----------------
X = data[["heart_rate", "spo2", "temperature", "hr_moving_avg"]]
y = data["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

joblib.dump(log_model, "models/logistic_model.pkl")
print("Logistic model saved.")

# ---------------- LSTM Model ----------------
hr = data["heart_rate"].values.reshape(-1, 1)

X_lstm, y_lstm = [], []
seq_len = 20

for i in range(len(hr) - seq_len - 10):
    X_lstm.append(hr[i:i+seq_len])
    y_lstm.append(hr[i+seq_len:i+seq_len+10])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(20, 1)),
    Dense(10)
])

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_lstm, y_lstm, epochs=5, batch_size=32, verbose=1)

model_lstm.save("models/lstm_model.h5")
print("LSTM model saved.")