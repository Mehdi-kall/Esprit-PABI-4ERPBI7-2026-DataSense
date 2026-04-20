# =============================================================================
# regression.py
# B2C Order Amount Prediction — Random Forest Regressor
# =============================================================================

import sys
import json
import warnings
import traceback
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── 1. LOAD MODEL ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
MODEL_PATH    = BASE_DIR / "model_regression.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns_regression.pkl"

try:
    model         = joblib.load(MODEL_PATH)
    expected_cols = joblib.load(FEATURES_PATH)
except FileNotFoundError as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)


# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # ── 1. Normalisation des noms
    df.columns = df.columns.str.strip()

    # ── 2. Features dérivées (avant tout drop)
    if "Montant_Remise" in df.columns and "prix_catalogue" in df.columns:
        df["remise_pct"] = (
            df["Montant_Remise"] /
            df["prix_catalogue"].replace(0, np.nan)
        ).fillna(0).clip(0, 1)

    if "Prix_Unité_moyen" in df.columns and "prix_catalogue" in df.columns:
        df["ratio_prix_catalogue"] = (
            df["Prix_Unité_moyen"] /
            df["prix_catalogue"].replace(0, np.nan)
        ).fillna(1).clip(0, 2)

    # ── 3. Forcer types AVANT reindex
    cat_cols = [
        "Numero_Commande",
        "Type_client",
        "ville_client",
        "canal_produit",
        "categorie",
        "type_vente",
        "type_paiement",
        "Trimestre",
        "Semestre",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    num_cols = [
        "Quantite_totale", "Prix_Unité_moyen", "Montant_Remise",
        "Montant_Livraison", "prix_catalogue", "En_Promo",
        "a_code_promo", "Annee", "Mois", "est_saison_haute",
        "remise_pct", "ratio_prix_catalogue",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── 4. Drop colonnes inutiles
    drop_cols = ["Montant_de_la_commande"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── 5. Alignement strict avec training
    df = df.reindex(columns=expected_cols, fill_value=0)

    # ── 6. Forcer Numero_Commande en string APRÈS reindex (fill_value=0 le met en int)
    if "Numero_Commande" in df.columns:
        df["Numero_Commande"] = df["Numero_Commande"].astype(str)

    return df


# ── 3. PREDICT ────────────────────────────────────────────────────────────────
def predict(payload: list[dict]) -> list[dict]:
    try:
        df_raw = pd.DataFrame(payload)

        if "Numero_Commande" in df_raw.columns:
            ids = df_raw["Numero_Commande"].astype(str).tolist()
        else:
            ids = list(range(len(df_raw)))

        X     = build_features(df_raw)
        preds = model.predict(X)

        return [
            {"commande_id": cid, "montant_predit": round(float(p), 2)}
            for cid, p in zip(ids, preds)
        ]

    except Exception as e:
        raise ValueError(traceback.format_exc())


# ── 4. ENTRY POINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    raw_input = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()

    try:
        payload = json.loads(raw_input)
        if isinstance(payload, dict):
            payload = [payload]
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    results = predict(payload)

    print(json.dumps(results, ensure_ascii=False, indent=2))

    pd.DataFrame(results).to_json(
        "predictions_regression.json",
        orient="records",
        force_ascii=False,
        indent=2
    )