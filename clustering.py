import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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

# 1. handle missing values
imputer = SimpleImputer(strategy="constant", fill_value=0)
X = imputer.fit_transform(df[CLUSTER_FEATURES])

# 2. scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. train model
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 4. save EVERYTHING needed for inference
df.to_csv("clustered_travel_dataset.csv", index=False)

print("Clustering completed successfully!")