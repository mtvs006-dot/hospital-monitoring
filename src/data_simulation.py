import numpy as np
import pandas as pd
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

np.random.seed(42)
samples = 1000

# Simulated biomedical signals
heart_rate = np.random.normal(80, 8, samples)
spo2 = np.random.normal(97, 1, samples)
temperature = np.random.normal(36.8, 0.3, samples)

# Inject abnormal conditions
heart_rate[200:220] = 140
spo2[500:520] = 85
temperature[700:710] = 39

data = pd.DataFrame({
    "heart_rate": heart_rate,
    "spo2": spo2,
    "temperature": temperature
})

# Moving average
data["hr_moving_avg"] = data["heart_rate"].rolling(5).mean()

# Threshold labeling
data["status"] = 0
data.loc[
    (data["heart_rate"] > 120) |
    (data["heart_rate"] < 50) |
    (data["spo2"] < 90) |
    (data["temperature"] > 38),
    "status"
] = 1

data.to_csv("data/patient_data.csv", index=False)

print("Dataset generated successfully.")
