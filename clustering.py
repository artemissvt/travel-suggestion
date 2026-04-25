import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("cleaned_travel_dataset.csv")

CLUSTER_FEATURES = [
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

df[CLUSTER_FEATURES] = df[CLUSTER_FEATURES].fillna(0)
X = df[CLUSTER_FEATURES]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 4  # good starting point for your dataset

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(X_scaled)


df.to_csv("clustered_travel_dataset.csv", index=False)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="constant", fill_value=0)

X = imputer.fit_transform(df[CLUSTER_FEATURES])
X_scaled = scaler.fit_transform(X)

print("Clustering completed!")
print(df[["Destination", "Cluster"]].head())
print("Saved as: clustered_travel_dataset.csv")