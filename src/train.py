"""Train, compare and save the churn pipeline.

Usage (from project root):
    python -m src.train
    python -m src.train --data data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

Flow: load -> clean -> split FIRST -> cross-validate candidates on the training
set only -> pick the best by CV ROC-AUC -> fit on training set -> evaluate ONCE
on the held-out test set -> save pipeline + metrics.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_DATA = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/churn_pipeline.joblib")
METRICS_PATH = Path("models/metrics.json")
FIG_PATH = Path("reports/figures/confusion_matrix.png")
RANDOM_STATE = 42

NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + ["Churn"]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    # TotalCharges has blank strings in the Kaggle file -> NaN (imputed in the pipeline)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print(f"Rows: {len(df)} | blank TotalCharges: {df['TotalCharges'].isna().sum()}")
    return df.drop_duplicates()


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def candidates() -> dict:
    return {
        "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
    }


def main(data_path: Path) -> None:
    df = load_data(data_path)
    y = df["Churn"].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Churn column must contain only 'Yes'/'No'.")
    X = df[FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores, pipes = {}, {}
    for name, model in candidates().items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        cv_scores[name], pipes[name] = float(score), pipe
        print(f"CV ROC-AUC  {name:<24} {score:.4f}")

    real = {k: v for k, v in cv_scores.items() if not k.startswith("baseline")}
    best = max(real, key=real.get)
    pipe = pipes[best].fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print(f"\nSelected: {best}\n")
    print(classification_report(y_test, pred, target_names=["No churn", "Churn"]))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:\n", cm)

    report = classification_report(y_test, pred, output_dict=True)
    metrics = {
        "model_version": f"{best}-v1",
        "selected_model": best,
        "cv_roc_auc": cv_scores,
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "test_pr_auc": float(average_precision_score(y_test, proba)),
        "test_accuracy": float(report["accuracy"]),
        "churn_precision": float(report["1"]["precision"]),
        "churn_recall": float(report["1"]["recall"]),
        "churn_f1": float(report["1"]["f1-score"]),
        "threshold": 0.5,
        "confusion_matrix": cm.tolist(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"]).plot(cmap="Blues")
    plt.title(f"Confusion matrix - {best}")
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    print(f"\nSaved {MODEL_PATH}, {METRICS_PATH}, {FIG_PATH}")
    print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    main(ap.parse_args().data)
