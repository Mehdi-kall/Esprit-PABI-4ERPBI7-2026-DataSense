# =============================================================================
# clustering.py
# Endpoint : POST /predict/clustering → segment client (Agglomerative n=5)
# Stratégie de scoring : nearest centroid (Agglomerative n'a pas predict())
# =============================================================================
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ── Chargement des artefacts ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

model    = joblib.load(BASE_DIR / "model_clustering.pkl")    # AgglomerativeClustering fitté
scaler   = joblib.load(BASE_DIR / "scaler_clustering.pkl")
features = joblib.load(BASE_DIR / "feature_columns_clustering.pkl")

# Centroïdes : moyenne des points par cluster dans l'espace scalé
# Calculés une seule fois au chargement depuis les labels du modèle
_labels    = model.labels_
_X_train   = None  # non disponible en prod → on utilise les centroïdes précalculés

SEGMENT_LABELS = {
    0: "B2C Dormants",
    1: "B2B Actifs Moyens",
    2: "B2C Remisés (à risque)",
    3: "B2B Grands Comptes",
    4: "B2C Récents Actifs",
}


def _nearest_centroid(X_scaled: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assigne chaque point au centroïde le plus proche (distance euclidienne)."""
    dists = np.linalg.norm(X_scaled[:, None, :] - centroids[None, :, :], axis=2)
    return dists.argmin(axis=1)


def predict(payload: list[dict]) -> list[dict]:
    """
    Prédit le segment de chaque client.

    Payload attendu : liste de dicts avec les 12 features :
        Recency, Frequency, Monetary, Avg_Basket, Total_Quantite,
        Nb_Produits, Total_Remise, Total_Remboursement, Nb_Ventes,
        Mode_Vente_Principal, Taux_Remise, Is_Entreprise

    Retourne : liste de dicts avec Cluster_ID et Segment_Label.
    """
    df = pd.DataFrame(payload)[features].fillna(0)
    X  = scaler.transform(df)

    # Centroïdes stockés dans le pkl (ajoutés lors du dump — voir note ci-dessous)
    centroids = model.centroids_  # attribut custom ajouté au moment du dump
    clusters  = _nearest_centroid(X, centroids)

    results = []
    for i, cluster in enumerate(clusters):
        results.append({
            **payload[i],
            "Cluster_ID":    int(cluster),
            "Segment_Label": SEGMENT_LABELS.get(int(cluster), "Inconnu"),
        })
    return results