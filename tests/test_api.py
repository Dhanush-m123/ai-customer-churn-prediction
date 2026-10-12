"""Run with: pytest -q
Trains a tiny pipeline on synthetic data, so the real model/CSV isn't needed."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app import database, main
from src.train import CATEGORICAL, FEATURES, NUMERIC, build_preprocessor

VALID = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 80.0, "TotalCharges": 160.0,
}


def _tiny_model():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({c: rng.choice(["Yes", "No"], n) for c in CATEGORICAL})
    df["Contract"] = rng.choice(["Month-to-month", "One year", "Two year"], n)
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 72, n)
    df["MonthlyCharges"] = rng.uniform(20, 120, n)
    df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]
    y = (df["tenure"] < 12).astype(int)
    pipe = Pipeline([("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=500))])
    return pipe.fit(df[FEATURES], y)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(main, "_model", _tiny_model())
    monkeypatch.setattr(main, "_model_version", "test-v1")
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {"Churn", "No Churn"}
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_blank_total_charges_is_allowed(client):
    payload = {k: v for k, v in VALID.items() if k != "TotalCharges"}
    assert client.post("/predict", json=payload).status_code == 200


def test_invalid_category_rejected(client):
    assert client.post("/predict", json={**VALID, "Contract": "Weekly"}).status_code == 422


def test_negative_charge_rejected(client):
    assert client.post("/predict", json={**VALID, "MonthlyCharges": -5}).status_code == 422


def test_history_records_prediction(client):
    client.post("/predict", json=VALID)
    rows = client.get("/history").json()
    assert len(rows) == 1 and rows[0]["contract"] == "Month-to-month"
