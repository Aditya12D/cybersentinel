"""Agent 1 - Threat Detection: 5-class flow classifier.

predict_threat(flow_record) -> {agent, predicted_class, confidence, probabilities}
Probabilities come straight from the trained model; confidence = max probability.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

CLASSES = ["BENIGN", "BRUTE_FORCE", "DOS", "DDOS", "PORT_SCAN"]  # stable output order
# Constant in training (zero information) or exact copies of another column.
DROP = [
    "Fwd URG Flags", "Bwd URG Flags", "URG Flag Count",
    "Fwd Header Length_duplicated_0", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Average Packet Size",
]
NON_FEATURES = {"record_id", "label"}


def feature_columns(columns) -> list[str]:
    return [c for c in columns if c not in NON_FEATURES and c not in DROP]


def prepare(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Same transform for training and inference: fixed column order, float32,
    +/-inf -> NaN (tree models route NaN natively). Missing columns -> NaN."""
    X = df.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    return X.astype("float32").replace([np.inf, -np.inf], np.nan)


@lru_cache(maxsize=1)
def _load():
    meta = json.loads((MODEL_DIR / "model_metadata.json").read_text())
    return joblib.load(MODEL_DIR / "threat_model.pkl"), meta


def predict_proba_frame(df: pd.DataFrame) -> np.ndarray:
    """(n, 5) probabilities in CLASSES order."""
    model, meta = _load()
    return model.predict_proba(prepare(df, meta["features"]))


def predict_batch(df: pd.DataFrame) -> list[dict]:
    return [_result(p) for p in predict_proba_frame(df)]


def predict_threat(flow_record: dict) -> dict:
    return _result(predict_proba_frame(pd.DataFrame([flow_record]))[0])


def _result(p) -> dict:
    i = int(np.argmax(p))
    return {
        "agent": "threat_detection",
        "predicted_class": CLASSES[i],
        "confidence": round(float(p[i]), 6),
        "probabilities": {c: round(float(v), 6) for c, v in zip(CLASSES, p)},
    }
