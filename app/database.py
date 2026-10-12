"""Tiny SQLite layer. All queries are parameterised."""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.getenv("CHURN_DB_PATH", "predictions.db")


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                contract TEXT,
                tenure INTEGER,
                monthly_charges REAL,
                prediction TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                model_version TEXT
            )"""
        )


def save_prediction(contract, tenure, monthly_charges, prediction, probability, model_version) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO predictions (created_at, contract, tenure, monthly_charges,"
            " prediction, churn_probability, model_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), contract, tenure,
             monthly_charges, prediction, probability, model_version),
        )


def recent_predictions(limit: int = 10) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
