from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load artifacts
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_model.pkl")
feature_cols: list = joblib.load("features.pkl")

# Load feature matrix and customer IDs separately, then join
df_features: pd.DataFrame = joblib.load("training_data.pkl")
customer_ids_series: pd.Series = joblib.load("customer_ids.pkl")

# Build a single lookup dataframe: customer_number + all features
df_model = df_features.copy()
df_model.insert(0, "customer_number", customer_ids_series.values)

cluster_map = {
    0: "Dormant / Low Engagement Users",
    1: "High-Value Digital Power Users",
    2: "Active Mass Market Users",
}

customer_ids = df_model["customer_number"].astype(str).unique().tolist()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "customer_ids": customer_ids, "result": None, "error": None, "selected": None},
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, customer_number: str = Form(...)):
    error = None
    result = None

    mask = df_model["customer_number"].astype(str) == customer_number

    if not mask.any():
        error = f"Customer '{customer_number}' not found in dataset."
    else:
        row = df_model[mask]

        # Use only the trained feature columns for prediction
        X = row[feature_cols]
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        cluster = int(kmeans.predict(X_pca)[0])

        # Build snapshot: feature name + rounded value
        snapshot = [
            {"Feature": col, "Value": round(float(row[col].values[0]), 4)}
            for col in feature_cols
        ]

        result = {
            "customer_number": customer_number,
            "cluster": cluster,
            "segment": cluster_map.get(cluster, "Unknown"),
            "snapshot": snapshot,
        }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "customer_ids": customer_ids,
            "result": result,
            "error": error,
            "selected": customer_number,
        },
    )
