# =============================================================================
# classification.py
# Loyalty prediction — XGBoost model
# Input  : JSON payload (from n8n / FastAPI)
# Output : JSON → stdout + predictions.json
# =============================================================================

import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 1. LOAD MODEL & FEATURE SCHEMA ────────────────────────────────────────────
MODEL_PATH    = "model_fidelite.pkl"
FEATURES_PATH = "feature_columns.pkl"

try:
    model         = joblib.load(MODEL_PATH)
    expected_cols = joblib.load(FEATURES_PATH)
except FileNotFoundError as e:
    print(json.dumps({"error": f"Model file not found: {e}"}))
    sys.exit(1)


# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduces the exact same feature engineering as the training notebook.
    Input  : raw DataFrame with original columns from DW query
    Output : feature DataFrame ready for prediction (no target column)
    """
    df = df_raw.copy()

    # Derived features
    df["ratio_saison_haute"] = (
        df["ca_saison_haute"] /
        df["montant_total"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    df["ratio_b2b"] = (
        (df["nb_factures_b2b"] * df["panier_moyen"]) /
        df["montant_total"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    df["score_rfm"] = (
        (1 / (df["recence_jours"] + 1)) *
        np.log1p(df["nb_factures_b2b"] + df["nb_commandes_b2c"]) *
        np.log1p(df["panier_moyen"])
    )

    df["mixte_b2b_b2c"] = (
        (df["nb_factures_b2b"] > 0) & (df["nb_commandes_b2c"] > 0)
    ).astype(int)

    # Drop columns excluded during training
    drop_cols = [
        "Client_PK", "est_fidele",
        "ca_saison_haute", "montant_total",
        "nb_commandes", "nb_factures_b2b", "nb_commandes_b2c",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # One-hot encoding
    cat_cols = [
        "Type_client", "ville_client",
        "paiement_dominant", "type_vente_dominant",
    ]
    cat_cols_present = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols_present, drop_first=True, dtype=int)

    # Fill missing numerics with 0 (median not available at inference — use 0)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)

    # Align columns with training schema
    df = df.reindex(columns=expected_cols, fill_value=0)

    return df


# ── 3. PREDICT ────────────────────────────────────────────────────────────────
def predict(payload: list[dict]) -> list[dict]:
    """
    payload : list of client records (raw, as extracted from DW)
    returns : list of {client_id, prediction, probabilite}
    """
    df_raw = pd.DataFrame(payload)

    # Keep client_id aside before feature engineering
    client_ids = df_raw["Client_PK"].tolist() if "Client_PK" in df_raw.columns else list(range(len(df_raw)))

    X = build_features(df_raw)

    predictions = model.predict(X)
    probas      = model.predict_proba(X)[:, 1]

    results = [
        {
            "client_id":   int(cid),
            "prediction":  int(pred),
            "probabilite": round(float(prob), 4),
        }
        for cid, pred, prob in zip(client_ids, predictions, probas)
    ]

    return results


# ── 4. ENTRY POINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Usage:
        python classification.py '[{...}, {...}]'
        python classification.py  (reads stdin if no arg)

    Output:
        - Prints JSON array to stdout  → consumed by n8n / FastAPI
        - Writes predictions.json      → optional file output
    """
    # Read input: CLI arg or stdin
    if len(sys.argv) > 1:
        raw_input = sys.argv[1]
    else:
        raw_input = sys.stdin.read()

    try:
        payload = json.loads(raw_input)
        if isinstance(payload, dict):
            payload = [payload]  # single record → wrap in list
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        sys.exit(1)

    results = predict(payload)

    # Stdout → n8n reads this
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # Optional file output
    pd.DataFrame(results).to_json(
        "predictions.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )