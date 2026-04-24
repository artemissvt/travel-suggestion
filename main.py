import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")


# =========================
# LOAD DATA
# =========================
DATA_PATH = "european_tour_destinations.csv"
def load_data(path):
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    print("\n📊 Dataset loaded!")
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    return df


# =========================
# PREPROCESSING
# =========================
def preprocess(df):
    df = df.copy()

    # Keep only numeric columns for ML simplicity
    numeric_df = df.select_dtypes(include=[np.number]).dropna()

    print("\n🧹 Using numeric columns for ML:")
    print(list(numeric_df.columns))

    return df, numeric_df


# =========================
# CLUSTERING (TRAVEL GROUPS)
# =========================
def clustering(numeric_df):
    scaler = StandardScaler()
    X = scaler.fit_transform(numeric_df)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    numeric_df = numeric_df.copy()
    numeric_df["Cluster"] = clusters

    print("\n🌍 Clustering complete!")
    print(numeric_df["Cluster"].value_counts())

    return numeric_df, kmeans, scaler


# =========================
# CLASSIFICATION (BUDGET LEVEL)
# =========================
def classification(df, numeric_df):
    # Try to find a cost-related column
    possible_cost_cols = [c for c in numeric_df.columns if "cost" in c.lower() or "price" in c.lower()]

    if not possible_cost_cols:
        print("\n⚠️ No cost column found → creating synthetic budget label")
        numeric_df["BudgetClass"] = pd.qcut(
            numeric_df.iloc[:, 0],
            q=3,
            labels=["Low", "Medium", "High"]
        )
        target_col = "BudgetClass"
    else:
        cost_col = possible_cost_cols[0]
        numeric_df["BudgetClass"] = pd.qcut(numeric_df[cost_col], q=3, labels=["Low", "Medium", "High"])
        target_col = "BudgetClass"

    X = numeric_df.drop(columns=["BudgetClass"])
    y = numeric_df["BudgetClass"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    print("\n🏷️ Classification model trained (Budget Level)")
    return model, X.columns, target_col


# =========================
# REGRESSION (PREDICT COST / SCORE)
# =========================
def regression(numeric_df):
    possible_targets = [c for c in numeric_df.columns if "cost" in c.lower() or "price" in c.lower() or "score" in c.lower()]

    if not possible_targets:
        print("\n⚠️ No cost/score column found → using first numeric column as target")
        target = numeric_df.columns[0]
    else:
        target = possible_targets[0]

    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    print(f"\n🔮 Regression model trained (predicting: {target})")
    return model, X.columns, target


# =========================
# RECOMMENDATION ENGINE
# =========================
def recommend(df, numeric_df, cluster_model, scaler):
    print("\n🧭 TRAVEL RECOMMENDER")

    try:
        budget = float(input("Enter max budget (numeric scale from dataset): "))
    except:
        budget = numeric_df.iloc[:, 0].mean()

    scaled = scaler.transform(numeric_df.drop(columns=["Cluster"], errors="ignore"))
    clusters = cluster_model.predict(scaled)

    numeric_df = numeric_df.copy()
    numeric_df["Cluster"] = clusters

    # Recommend "best cluster"
    best_cluster = numeric_df.groupby("Cluster").mean().iloc[:, 0].idxmax()

    results = numeric_df[numeric_df["Cluster"] == best_cluster].head(5)

    print("\n✨ Top recommended destinations (cluster-based):")
    print(results)


# =========================
# MAIN MENU

def main():
    print("📂 Loading dataset from:", DATA_PATH)

    df = load_data(DATA_PATH)
    df, numeric_df = preprocess(df)

    numeric_df, cluster_model, scaler = clustering(numeric_df)

    clf_model, clf_features, clf_target = classification(df, numeric_df)
    reg_model, reg_features, reg_target = regression(numeric_df)

    while True:
        print("\n==============================")
        print("🌍 EU TRAVEL AI MENU")
        print("1. Show clusters")
        print("2. Predict budget class (demo)")
        print("3. Predict numeric value (demo)")
        print("4. Recommend destinations")
        print("5. Exit")

        choice = input("Choose: ")

        if choice == "1":
            print(numeric_df[["Cluster"]].head(10))

        elif choice == "2":
            sample = numeric_df.drop(columns=["Cluster"]).iloc[0:1]
            pred = clf_model.predict(sample)[0]
            print("🏷️ Predicted budget class:", pred)

        elif choice == "3":
            sample = numeric_df.drop(columns=["Cluster"]).iloc[0:1]
            pred = reg_model.predict(sample)[0]
            print("🔮 Predicted value:", pred)

        elif choice == "4":
            recommend(df, numeric_df, cluster_model, scaler)

        elif choice == "5":
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()