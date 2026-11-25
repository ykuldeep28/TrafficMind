import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "historical_traffic_dataset.csv"
MODEL_PATH = DATA_DIR / "traffic_ml_model.joblib"

ROAD_MAP = {"none": 0, "N": 1, "S": 2, "E": 3, "W": 4}

def prepare_features(df: pd.DataFrame) -> np.ndarray:
    n = df["north"].values
    s = df["south"].values
    e = df["east"].values
    w = df["west"].values
    
    total = np.maximum(1, n + s + e + w)
    ns_ratio = (n + s) / total
    ew_ratio = (e + w) / total
    
    closed_series = df["closed_road"].astype(str).str.upper()
    em_series = df["emergency_approach"].astype(str).str.upper()

    closed_code = closed_series.map(lambda x: ROAD_MAP.get(x, 0)).values
    emergency_code = em_series.map(lambda x: ROAD_MAP.get(x, 0)).values

    is_closed_ew = closed_series.isin(["E", "W"]).astype(float).values
    is_closed_ns = closed_series.isin(["N", "S"]).astype(float).values
    is_em_ns = em_series.isin(["N", "S"]).astype(float).values
    is_em_ew = em_series.isin(["E", "W"]).astype(float).values

    X = np.column_stack([
        n, s, e, w, total, ns_ratio, ew_ratio,
        closed_code, emergency_code,
        is_closed_ew, is_closed_ns, is_em_ns, is_em_ew
    ])
    return X


def train():
    print(f"Loading historical traffic dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Dataset shape: {df.shape}")

    X = prepare_features(df)
    y = df[["optimal_ns_green", "optimal_ew_green"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Multi-Output Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae_ns = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_ew = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    r2_ns = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_ew = r2_score(y_test[:, 1], y_pred[:, 1])

    print("\n--- Model Training & Validation Results ---")
    print(f"NS Green - MAE: {mae_ns:.2f}s | R²: {r2_ns:.3f}")
    print(f"EW Green - MAE: {mae_ew:.2f}s | R²: {r2_ew:.3f}")
    print("------------------------------------------")

    joblib.dump({
        "model": model,
        "feature_names": [
            "north", "south", "east", "west", "total",
            "ns_ratio", "ew_ratio", "closed_code", "emergency_code",
            "is_closed_ew", "is_closed_ns", "is_em_ns", "is_em_ew"
        ],
        "metrics": {
            "mae_ns": round(mae_ns, 2),
            "mae_ew": round(mae_ew, 2),
            "r2_ns": round(r2_ns, 3),
            "r2_ew": round(r2_ew, 3),
            "training_samples": len(X_train),
        }
    }, MODEL_PATH)
    print(f"✅ Successfully exported trained model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
