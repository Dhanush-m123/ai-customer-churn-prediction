import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app import database
from app.schemas import CustomerInput, PredictionOut
from src.train import FEATURES

log = logging.getLogger("churn")

MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", "models/churn_pipeline.joblib"))
METRICS_PATH = Path(os.getenv("CHURN_METRICS_PATH", "models/metrics.json"))
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))

_model = None
_model_version = "unknown"


def get_model():
    """Load the saved pipeline once; return None if training hasn't been run yet."""
    global _model, _model_version
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        if METRICS_PATH.exists():
            _model_version = json.loads(METRICS_PATH.read_text()).get("model_version", "v1")
        else:
            _model_version = "v1"
    return _model


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.init_db()
    get_model()
    yield


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_model() is not None}


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "No metrics found. Run: python -m src.train")
    return json.loads(METRICS_PATH.read_text())


@app.post("/predict", response_model=PredictionOut)
def predict(customer: CustomerInput):
    model = get_model()
    if model is None:
        raise HTTPException(503, "Model not trained yet. Run: python -m src.train")
    try:
        row = pd.DataFrame([customer.model_dump()])[FEATURES]
        row["TotalCharges"] = pd.to_numeric(row["TotalCharges"], errors="coerce")
        probability = float(model.predict_proba(row)[0][1])
    except Exception:
        log.exception("Prediction failed")
        raise HTTPException(500, "Prediction failed. Check the server logs.")

    label = "Churn" if probability >= THRESHOLD else "No Churn"
    database.save_prediction(customer.Contract, customer.tenure, customer.MonthlyCharges,
                             label, round(probability, 4), _model_version)
    return PredictionOut(prediction=label, churn_probability=round(probability, 4),
                         threshold=THRESHOLD, model_version=_model_version)


@app.get("/history")
def history(limit: int = 10):
    return database.recent_predictions(max(1, min(limit, 100)))


# Serve the web UI last so it doesn't shadow the API routes.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
