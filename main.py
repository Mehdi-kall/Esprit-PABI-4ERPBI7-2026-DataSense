# =============================================================================
# main.py
# FastAPI — ML Prediction API
# Endpoints :
#   POST /run/etl                 → Lance les jobs Talend (via conteneur Linux)
#   POST /predict/classification  → loyalty prediction (XGBoost)
#   POST /predict/regression      → order amount prediction (Random Forest)
#   POST /predict/timeseries      → monthly revenue forecast (XGBoost)
#   POST /predict/clustering      → customer segment (Agglomerative n=5)
#
# Run : uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# =============================================================================
import sys
import time
import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

# ── Import prediction functions from each module ──────────────────────────────
from classification import predict as predict_classification
from regression     import predict as predict_regression
from timeseries     import predict as predict_timeseries
from clustering     import predict as predict_clustering

# ── Config ────────────────────────────────────────────────────────────────────
TALEND_URL = "http://localhost:5001/run/etl"   # conteneur talend exposé sur le port 5001

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("api.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="E-Commerce ML API",
    description="ETL Talend + Prédictions ML : fidélité client, montant commande, CA mensuel, segmentation client",
    version="1.0.0"
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    data: list[dict[str, Any]]


class PredictResponse(BaseModel):
    model:        str
    n_records:    int
    duration_ms:  float
    predictions:  list[dict[str, Any]]


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Helper ML ─────────────────────────────────────────────────────────────────
def run_prediction(model_name: str, predict_fn, payload: list[dict]) -> PredictResponse:
    logger.info(f"[{model_name}] Requête reçue — {len(payload)} enregistrements")
    start = time.time()

    try:
        results = predict_fn(payload)
    except Exception as e:
        logger.error(f"[{model_name}] Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

    duration = round((time.time() - start) * 1000, 2)
    logger.info(f"[{model_name}] OK — {len(results)} prédictions en {duration} ms")

    return PredictResponse(
        model=model_name,
        n_records=len(results),
        duration_ms=duration,
        predictions=results
    )


# ── ETL Endpoint ──────────────────────────────────────────────────────────────
@app.post("/run/etl")
def run_etl():
    """
    Lance tous les jobs Talend dans l'ordre (dimensions → faits).
    Délègue au conteneur Linux talend_runner via HTTP.
    """
    logger.info("[ETL] Lancement des jobs Talend")
    start = time.time()
    try:
        response = httpx.post(TALEND_URL, timeout=600)
        result   = response.json()
        duration = round((time.time() - start) * 1000, 2)
        logger.info(f"[ETL] Terminé en {duration} ms — {result.get('failed', '?')} échecs")
        return result
    except httpx.ConnectError:
        logger.error("[ETL] Impossible de joindre le conteneur talend")
        raise HTTPException(status_code=503, detail="Conteneur Talend inaccessible — vérifier que n8n-project-talend-1 tourne")
    except Exception as e:
        logger.error(f"[ETL] Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── ML Endpoints ──────────────────────────────────────────────────────────────
@app.post("/predict/classification", response_model=PredictResponse)
def classification_endpoint(request: PredictRequest):
    """
    Prédit la fidélité client (0 = non fidèle, 1 = fidèle).
    Payload : liste de clients avec leurs features brutes.
    """
    return run_prediction("classification", predict_classification, request.data)


@app.post("/predict/regression", response_model=PredictResponse)
def regression_endpoint(request: PredictRequest):
    """
    Prédit le montant d'une commande B2C (en TND).
    Payload : liste de commandes avec leurs features brutes.
    """
    return run_prediction("regression", predict_regression, request.data)


@app.post("/predict/timeseries", response_model=PredictResponse)
def timeseries_endpoint(request: PredictRequest):
    """
    Prédit le CA mensuel (XGBoost sur lags).
    Payload : historique mensuel (minimum 4 mois) avec CA_Mensuel renseigné.
    """
    return run_prediction("timeseries", predict_timeseries, request.data)


@app.post("/predict/clustering", response_model=PredictResponse)
def clustering_endpoint(request: PredictRequest):
    """
    Assigne chaque client à un segment (Agglomerative n=5, nearest centroid).
    Payload : liste de clients avec les 12 features RFM + comportementales.
    Retourne : Cluster_ID (0-4) + Segment_Label métier.
    """
    return run_prediction("clustering", predict_clustering, request.data)