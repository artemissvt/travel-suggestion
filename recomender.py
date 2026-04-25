import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD CLUSTERED DATASET
# =========================

df = pd.read_csv("clustered_travel_dataset.csv")

# =========================
# FEATURES USED FOR MATCHING USER → CLUSTER
# =========================

FEATURES = [
    "Spring",
    "Summer",
    "Autumn",
    "Winter",
    "Cost_of_Living_Encoded",
    "Historic",
    "Medieval",
    "Beach",
    "Architecture",
    "Capital"
]

# =========================
# USER INPUT PARSER
# =========================

def parse_user_input(text):
    text = text.lower()

    prefs = {
        "Spring": 0,
        "Summer": 0,
        "Autumn": 0,
        "Winter": 0,
        "Historic": 0,
        "Medieval": 0,
        "Beach": 0,
        "Architecture": 0,
        "Capital": 0,
        "Cost_of_Living_Encoded": 1,
    }

    if "spring" in text:
        prefs["Spring"] = 1
    if "summer" in text:
        prefs["Summer"] = 1
    if "autumn" in text or "fall" in text:
        prefs["Autumn"] = 1
    if "winter" in text:
        prefs["Winter"] = 1

    if "historic" in text:
        prefs["Historic"] = 1
    if "medieval" in text:
        prefs["Medieval"] = 1
    if "beach" in text or "island" in text:
        prefs["Beach"] = 1
    if "architecture" in text:
        prefs["Architecture"] = 1

    if "cheap" in text or "budget" in text:
        prefs["Cost_of_Living_Encoded"] = 0
    elif "expensive" in text or "luxury" in text:
        prefs["Cost_of_Living_Encoded"] = 3

    return prefs


# =========================
# FIND BEST CLUSTER
# =========================

def find_best_cluster(user_vector):
    """
    Instead of training a new model,
    we match user to clusters by similarity.
    """

    cluster_scores = {}

    for cluster_id in df["Cluster"].unique():
        cluster_data = df[df["Cluster"] == cluster_id]

        cluster_center = cluster_data[FEATURES].mean().values

        distance = np.sqrt(np.sum((user_vector - cluster_center) ** 2))

        cluster_scores[cluster_id] = distance

    # smallest distance = best cluster
    best_cluster = min(cluster_scores, key=cluster_scores.get)

    return best_cluster


# =========================
# RECOMMEND INSIDE CLUSTER
# =========================

def recommend(user_text, top_n=5):

    prefs = parse_user_input(user_text)

    user_vector = np.array([prefs[f] for f in FEATURES])

    # STEP 1: find best cluster
    cluster = find_best_cluster(user_vector)

    print(f"\n🔍 Matched Cluster: {cluster}")

    # STEP 2: filter dataset to that cluster
    cluster_df = df[df["Cluster"] == cluster].copy()

    # STEP 3: compute simple similarity inside cluster
    distances = []

    for _, row in cluster_df.iterrows():
        v = row[FEATURES].values.astype(float)

        dist = np.sqrt(np.sum((user_vector - v) ** 2))
        distances.append(dist)

    cluster_df["Score"] = distances

    results = cluster_df.sort_values("Score").head(top_n)

    return results


# =========================
# TERMINAL INTERFACE
# =========================

print("=" * 60)
print("CLUSTER-BASED TRAVEL RECOMMENDER (ML VERSION)")
print("=" * 60)
print()

user_input = input("Describe your ideal destination: ")

results = recommend(user_input)

print("\n" + "=" * 60)
print("TOP RECOMMENDATIONS")
print("=" * 60)

for i, row in enumerate(results.iterrows(), 1):
    data = row[1]
    print(f"{i}. {data['Destination']} ({data['Country']})")

print("\nDone.")