import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

df = pd.read_csv("cleaned_travel_dataset.csv")

FEATURES = [
    "Spring","Summer","Autumn","Winter",
    "Cost_of_Living_Encoded",
    "Historic","Medieval","Beach","Architecture","Capital"
]

imputer = SimpleImputer(strategy="constant", fill_value=0)
X = imputer.fit_transform(df[FEATURES])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
df.to_csv("clustered_travel_dataset.csv", index=False)

print("Training complete")