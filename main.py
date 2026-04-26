import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("european_tour_destinations.csv", encoding="latin1")

# tourists
def convert_tourists(value):
    """
    Convert values like:
    '14 million' -> 14000000
    '35-40 million' -> 37500000
    '200,000' -> 200000
    """

    if pd.isna(value):
        return np.nan

    value = str(value).lower().strip()

    # Remove commas
    value = value.replace(",", "")

    # Handle values with "million"
    if "million" in value:
        value = value.replace("million", "").strip()

        # Handle ranges like 35-40
        if "-" in value:
            parts = value.split("-")
            try:
                numbers = [float(part.strip()) for part in parts]
                return np.mean(numbers) * 1_000_000
            except:
                return np.nan

        # Normal single value like 14
        try:
            return float(value) * 1_000_000
        except:
            return np.nan

    # Handle normal numeric values like 200000
    try:
        return float(value)
    except:
        return np.nan

# Convert to numeric
df["Annual_Tourists"] = df["Approximate Annual Tourists"].apply(convert_tourists)

# Log transformation (recommended)
df["Annual_Tourists_Log"] = np.log1p(df["Annual_Tourists"])

# Standard scaling
scaler = StandardScaler()
df["Annual_Tourists_Scaled"] = scaler.fit_transform(
    df[["Annual_Tourists_Log"]]
)


df["Best Time to Visit"] = df["Best Time to Visit"].fillna("").str.lower()

df["Spring"] = df["Best Time to Visit"].str.contains(
    "spring|april|may", case=False, na=False
).astype(int)

df["Summer"] = df["Best Time to Visit"].str.contains(
    "summer|june|july|august", case=False, na=False
).astype(int)

df["Autumn"] = df["Best Time to Visit"].str.contains(
    "autumn|fall|sept|oct|nov", case=False, na=False
).astype(int)

df["Winter"] = df["Best Time to Visit"].str.contains(
    "winter|dec|jan|feb", case=False, na=False
).astype(int)


cost_mapping = {
    "low": 0,
    "medium": 1,
    "medium-high": 2,
    "high": 3,
    "very high": 4,
}

df["Cost_of_Living_Encoded"] = (
    df["Cost of Living"]
    .str.lower()
    .str.strip()
    .map(cost_mapping)
)


df["Historic"] = df["Cultural Significance"].str.contains(
    "historic|historical", case=False, na=False
).astype(int)

df["Medieval"] = df["Cultural Significance"].str.contains(
    "medieval", case=False, na=False
).astype(int)

df["Beach"] = df["Cultural Significance"].str.contains(
    "beach|coast|island", case=False, na=False
).astype(int)

df["Architecture"] = df["Cultural Significance"].str.contains(
    "architecture", case=False, na=False
).astype(int)

df["Capital"] = df["Cultural Significance"].str.contains(
    "capital", case=False, na=False
).astype(int)


features_for_clustering = [
    "Annual_Tourists_Scaled",
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

final_df = df[
    ["Destination", "Country"] + features_for_clustering
]


final_df.to_csv("cleaned_travel_dataset.csv", index=False)

"""
print("Dataset cleaned successfully.")
print()
print(final_df.head())
print()
print("Saved as: cleaned_travel_dataset.csv")
"""
