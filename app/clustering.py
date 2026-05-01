import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "cleaned_travel_dataset.csv")

FEATURES = [
    "Spring", "Summer", "Autumn", "Winter",
    "Cost_of_Living_Encoded",
    "Historic", "Medieval", "Beach", "Architecture", "Capital"
]

imputer = SimpleImputer(strategy="constant", fill_value=0)
X = imputer.fit_transform(df[FEATURES])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Save outputs inside app/ so Docker picks them up
models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(kmeans, models_dir / "kmeans_model.pkl")
joblib.dump(scaler, models_dir / "scaler.pkl")
df.to_csv(BASE_DIR / "clustered_travel_dataset.csv", index=False)

print("Training complete.")
print(f"Models saved to: {models_dir}")
print(f"Clustered dataset saved to: {BASE_DIR / 'clustered_travel_dataset.csv'}")