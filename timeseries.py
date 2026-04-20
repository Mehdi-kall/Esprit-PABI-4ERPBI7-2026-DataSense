# =============================================================================
# timeseries.py
# Monthly Revenue Forecasting — XGBoost Model
# Input  : JSON payload (list of monthly aggregated records)
# Output : JSON → stdout + predictions_timeseries.json
# =============================================================================

import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 1. LOAD MODEL ─────────────────────────────────────────────────────────────
from pathlib import Path
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_timeseries.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError as e:
    print(json.dumps({"error": f"Model file not found: {e}"}))
    sys.exit(1)


# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduces the exact feature engineering from the training notebook.

    Expected input columns (monthly aggregated):
        - Date             : YYYY-MM-DD (first day of month)
        - CA_Mensuel       : monthly revenue (target — used to build lags)
        - Quantite_Mensuelle
        - Nb_Commandes
        - Remise_Totale
        - Prix_Moyen_Unite

    The function builds lag + rolling features, then drops rows with NaN
    (same as training). Returns X ready for prediction.
    """
    df = df_raw.copy()

    # Ensure Date index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

    # Lag features (identical to training)
    df["lag_1"] = df["CA_Mensuel"].shift(1)
    df["lag_2"] = df["CA_Mensuel"].shift(2)
    df["lag_3"] = df["CA_Mensuel"].shift(3)
    df["rolling_mean_3"] = df["CA_Mensuel"].rolling(3).mean()

    # Drop NaN rows introduced by lags (same as training .dropna())
    df = df.dropna()

    # Features used at training — drop target
    X = df.drop(columns=["CA_Mensuel"])

    return X, df.index


# ── 3. PREDICT ────────────────────────────────────────────────────────────────
def predict(payload: list[dict]) -> list[dict]:
    """
    payload : list of monthly records (aggregated, as built from DW)
    returns : list of {date, ca_predit}

    Note: at least 3 months of history required to compute lags.
    """
    df_raw = pd.DataFrame(payload)

    if len(df_raw) < 4:
        return [{"error": "Minimum 4 months of history required to compute lag features."}]

    X, dates = build_features(df_raw)

    predictions = model.predict(X)

    results = [
        {
            "date":      str(date.date()),
            "ca_predit": round(float(pred), 2),
        }
        for date, pred in zip(dates, predictions)
    ]

    return results


# ── 4. ENTRY POINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Usage:
        python timeseries.py '[{...}, {...}]'
        python timeseries.py  (reads stdin if no arg)

    Minimum payload: 4 monthly records with CA_Mensuel populated.

    Output:
        - Prints JSON array to stdout  → consumed by n8n / FastAPI
        - Writes predictions_timeseries.json
    """
    if len(sys.argv) > 1:
        raw_input = sys.argv[1]
    else:
        raw_input = sys.stdin.read()

    try:
        payload = json.loads(raw_input)
        if isinstance(payload, dict):
            payload = [payload]
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        sys.exit(1)

    results = predict(payload)

    print(json.dumps(results, ensure_ascii=False, indent=2))

    pd.DataFrame(results).to_json(
        "predictions_timeseries.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )