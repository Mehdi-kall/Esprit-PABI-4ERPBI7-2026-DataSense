from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# ----------------------------
# INIT APP
# ----------------------------
app = FastAPI(title="Customer Segmentation API")

load_dotenv()

# ----------------------------
# DB CONNECTION
# ----------------------------
engine = create_engine(
    f"mssql+pyodbc://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_SERVER')}/{os.getenv('DB_NAME')}"
    f"?driver=ODBC+Driver+17+for+SQL+Server"
)

# ----------------------------
# FEATURES
# ----------------------------
FEATURES = [
    'Recency','Frequency','Monetary','Avg_Basket','Total_Quantite',
    'Nb_Produits','Total_Remise','Total_Remboursement',
    'Nb_Ventes','Mode_Vente_Principal','Taux_Remise','Is_Entreprise'
]

SEGMENT_NAMES = {
    0: "Clients standards ",
    1: "Entreprises actives (B2B)",
    2: "Chasseurs de promos",
    3: "Entreprises stratégiques à risque ",
    4: "Clients fidèles actifs"
}

QUALITY_THRESHOLD = 0.25

# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ----------------------------
# LOAD DATA
# ----------------------------
def load_data():
    df = pd.read_sql('SELECT * FROM Fact_Ventes', engine)
    clients = pd.read_sql('SELECT * FROM Dim_Client', engine)
    dates = pd.read_sql('SELECT * FROM DimDates', engine)
    return df, clients, dates

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def build_dataset(df, clients, dates):

    df_full = df.merge(clients, left_on='Client_FK', right_on='Client_PK', how='left')
    df_full = df_full.merge(dates, left_on='Date_FK', right_on='Date_PK', how='left')

    df_full['Date'] = pd.to_datetime(df_full['Date'])
    today = df_full['Date'].max()

    rfm = df_full.groupby('Client_FK').agg(
        Recency=('Date', lambda x: (today - x.max()).days),
        Frequency=('Numero_Commande', 'nunique'),
        Monetary=('Montant_de_la_commande', 'sum')
    )

    extra = df_full.groupby('Client_FK').agg(
        Avg_Basket=('Montant_de_la_commande', 'mean'),
        Total_Quantite=('Quantite', 'sum'),
        Nb_Produits=('Produit_FK', 'nunique'),
        Total_Remise=('Montant_Remise', 'sum'),
        Total_Remboursement=('Montant_Remboursement', 'sum'),
        Nb_Ventes=('Ventes_PK', 'count'),
        Mode_Vente_Principal=('Type_Vente_FK', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
    )

    # sécurité division
    rfm['Monetary'] = rfm['Monetary'].replace(0, np.nan)
    extra['Taux_Remise'] = extra['Total_Remise'] / rfm['Monetary']
    extra['Taux_Remise'] = extra['Taux_Remise'].fillna(0)

    dataset = rfm.merge(extra, on='Client_FK')

    client_type = clients[['Client_PK','Type_client']].rename(columns={'Client_PK':'Client_FK'})
    dataset = dataset.merge(client_type, on='Client_FK', how='left')

    dataset['Is_Entreprise'] = (dataset['Type_client'] == 'Entreprise').astype(int)

    dataset = dataset.drop(columns=['Type_client']).fillna(0)

    return dataset

# ----------------------------
# CLUSTERING + QUALITY SCORE
# ----------------------------
def apply_model(dataset):

    scaler = StandardScaler()
    scaled = scaler.fit_transform(dataset[FEATURES])

    model = AgglomerativeClustering(n_clusters=5, linkage='ward')
    clusters = model.fit_predict(scaled)

    dataset['Cluster'] = clusters
    dataset['Segment'] = dataset['Cluster'].map(SEGMENT_NAMES)

    # ----------------------------
    # QUALITY SCORE (SEUIL)
    # ----------------------------
    score = silhouette_score(scaled, clusters)

    return dataset, score

# ----------------------------
# MAIN API
# ----------------------------
@app.post("/predict")
def predict():

    try:
        df, clients, dates = load_data()
        dataset = build_dataset(df, clients, dates)

        dataset, score = apply_model(dataset)

        # résumé segments
        result = dataset[['Cluster','Segment']].value_counts().reset_index()
        result.columns = ['Cluster', 'Segment', 'count']

        quality_status = "OK" if score >= QUALITY_THRESHOLD else "LOW_QUALITY"

        return {
            "nb_clients": len(dataset),
            "silhouette_score": float(score),
            "quality_threshold": QUALITY_THRESHOLD,
            "quality_status": quality_status,
            "segments": result.to_dict(orient="records")
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }