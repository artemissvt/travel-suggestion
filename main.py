from encodings import latin_1

import pandas as pd
from sklearn.cluster import KMeans

# Load dataset
try:
    df = pd.read_csv("european_tour_destinations.csv", encoding="latin1")
except FileNotFoundError:
    print("Dataset file not found. Make sure it's in the project folder.")
    exit()

# Clean dataset
df = df.dropna()

print("Columns in dataset:", df.columns)

# Adjust column names if needed
city_col = "City" if "City" in df.columns else df.columns[0]
country_col = "Country" if "Country" in df.columns else df.columns[1]

# Simulated temperature scoring
def assign_temp(country):
    warm = ["Spain", "Italy", "Greece", "Portugal"]
    cold = ["Norway", "Sweden", "Finland"]

    if country in warm:
        return 30
    elif country in cold:
        return 10
    else:
        return 20

df["temp_score"] = df[country_col].apply(assign_temp)

# Features for ML
features = df[["temp_score"]]

# Train model
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(features)

# Recommendation function
def recommend_destinations(preference="warm", n=5):
    if preference == "warm":
        target = 30
    elif preference == "cold":
        target = 10
    else:
        target = 20

    df["distance"] = abs(df["temp_score"] - target)
    results = df.sort_values(by="distance").head(n)

    return results[[city_col, country_col]]

# Run test
if __name__ == "__main__":
    print("\nRecommended destinations:\n")

    recs = recommend_destinations("warm", 5)

    for _, row in recs.iterrows():
        print(f"{row[city_col]}, {row[country_col]}")