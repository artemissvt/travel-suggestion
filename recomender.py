import pandas as pd
import numpy as np

df = pd.read_csv("clustered_travel_dataset.csv")

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

# IMPORTANT: use SAME preprocessing style as clustering
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

    if "cheap" in text: prefs["Cost_of_Living_Encoded"] = 0
    if "luxury" in text: prefs["Cost_of_Living_Encoded"] = 3

    return prefs


def recommend(user_text, top_n=5):

    prefs = parse_user_input(user_text)
    user_vector = np.array([prefs[f] for f in FEATURES])

    # STEP 1: find best cluster
    cluster_scores = {}

    for c in df["Cluster"].unique():
        cluster_data = df[df["Cluster"] == c]
        cluster_center = cluster_data[FEATURES].mean().values

        dist = np.linalg.norm(user_vector - cluster_center)
        cluster_scores[c] = dist

    best_cluster = min(cluster_scores, key=cluster_scores.get)

    print(f"\nMatched Cluster: {best_cluster}")

    # STEP 2: filter cluster
    cluster_df = df[df["Cluster"] == best_cluster]

    # STEP 3: rank inside cluster
    scores = np.linalg.norm(
        cluster_df[FEATURES].values - user_vector,
        axis=1
    )

    cluster_df = cluster_df.copy()
    cluster_df["Score"] = scores

    return cluster_df.sort_values("Score").head(top_n)


#just for testing

"""
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
"""