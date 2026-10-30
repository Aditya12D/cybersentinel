"""Binary anomaly metrics (BENIGN = 0, any attack = 1) used by calibration and evaluation."""
from __future__ import annotations

import numpy as np

from . import config


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y)); tn = int(np.sum(~pred & ~y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def per_class_detection(labels: np.ndarray, pred: np.ndarray) -> dict:
    """Detection rate per attack class and flag rate for BENIGN (= FPR)."""
    out = {}
    for lab in (config.BENIGN_LABEL, *config.ATTACK_LABELS):
        m = labels == lab
        out[lab] = {"n": int(m.sum()), "flagged": int(pred[m].sum()),
                    "flag_rate": float(pred[m].mean()) if m.any() else 0.0}
    return out
