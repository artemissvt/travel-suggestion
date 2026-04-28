import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("clustered_travel_dataset.csv")

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = [
    "Spring","Summer","Autumn","Winter",
    "Cost_of_Living_Encoded",
    "Historic","Medieval","Beach","Architecture","Capital"
]

def parse_user_input(text):
    text = text.lower()

    prefs = dict.fromkeys(FEATURES, 0)
    prefs["Cost_of_Living_Encoded"] = 1

    if "spring" in text: prefs["Spring"] = 1
    if "summer" in text: prefs["Summer"] = 1
    if "autumn" in text or "fall" in text: prefs["Autumn"] = 1
    if "winter" in text: prefs["Winter"] = 1

    if "historic" in text: prefs["Historic"] = 1
    if "medieval" in text: prefs["Medieval"] = 1
    if "beach" in text or "island" in text: prefs["Beach"] = 1
    if "architecture" in text: prefs["Architecture"] = 1
    if "capital" in text: prefs["capital"] = 1

    if "cheap" in text or "budget" in text: prefs["Cost_of_Living_Encoded"] = 0
    if "medium" in text: prefs["Cost_of_Living_Encoded"] = 1
    if "luxury" in text: prefs["Cost_of_Living_Encoded"] = 3

    return prefs


def recommend(user_text, top_n=5):

    prefs = parse_user_input(user_text)
    user_vector = np.array([prefs[f] for f in FEATURES]).reshape(1, -1)

    user_scaled = scaler.transform(user_vector)

    # predict cluster using trained model
    best_cluster = kmeans.predict(user_scaled)[0]

    cluster_df = df[df["Cluster"] == best_cluster]

    scores = np.linalg.norm(
        cluster_df[FEATURES].values - user_vector,
        axis=1
    )

    cluster_df = cluster_df.copy()
    cluster_df["Score"] = scores

    return cluster_df.sort_values("Score").head(top_n)