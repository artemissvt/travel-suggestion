import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "clustered_travel_dataset.csv")
kmeans = joblib.load(BASE_DIR / "models" / "kmeans_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

FEATURES = [
    "Spring", "Summer", "Autumn", "Winter",
    "Cost_of_Living_Encoded",
    "Historic", "Medieval", "Beach", "Architecture", "Capital"
]


def parse_user_input(text):
    text = text.lower()
    prefs = dict.fromkeys(FEATURES, 0)
    prefs["Cost_of_Living_Encoded"] = 1  # default: medium cost

    if "spring" in text or "sunny" in text:
        prefs["Spring"] = 1
    if "summer" in text or "sunny" in text:
        prefs["Summer"] = 1
    if "autumn" in text or "fall" in text:
        prefs["Autumn"] = 1
    if "winter" in text:
        prefs["Winter"] = 1
    if "historic" in text:
        prefs["Historic"] = 2
    if "medieval" in text:
        prefs["Medieval"] = 2
    if "beach" in text or "island" in text:
        prefs["Beach"] = 2
    if "architecture" in text:
        prefs["Architecture"] = 2
    if "capital" in text:
        prefs["Capital"] = 3
    if "cheap" in text or "budget" in text:
        prefs["Cost_of_Living_Encoded"] = 0
    if "luxury" in text:
        prefs["Cost_of_Living_Encoded"] = 3

    return prefs


def recommend(user_text, top_n=5):
    prefs = parse_user_input(user_text)
    user_vector = np.array([prefs[f] for f in FEATURES]).reshape(1, -1)

    # Scale user input the same way the model was trained
    user_scaled = scaler.transform(user_vector)

    # Find the closest cluster
    cluster = kmeans.predict(user_scaled)[0]

    # Score destinations within that cluster by distance to user preferences
    cluster_df = df[df["Cluster"] == cluster].copy()
    scores = np.linalg.norm(
        cluster_df[FEATURES].values - user_vector,
        axis=1
    )
    cluster_df["Score"] = scores
    return cluster_df.sort_values("Score").head(top_n)