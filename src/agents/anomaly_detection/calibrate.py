"""Phase 5: threshold calibration on the eval *calibration* half.

Methodology
-----------
* Sweep thresholds t on the normalised anomaly score (benign percentile).
* For each t record precision, recall, F1, benign FPR, accuracy and per-class
  detection rates -> outputs/threshold_analysis.csv.
* Selection rule (``config.CALIBRATION_MAX_BENIGN_FPR``): maximise F1 subject
  to benign FPR <= 3%. Accuracy is never optimised. The FPR budget makes the
  choice robust to the eval partition's unrealistic ~47% attack prevalence
  (F1 alone would happily trade many more false alarms for recall).
  Ties are broken towards the higher (more conservative) threshold.
* Two alternative operating points are stored for Agent 3:
  ``strict`` (FPR <= 1%) and ``sensitive`` (FPR <= 5%), same rule.
* The holdout half is not touched here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .metrics import binary_metrics

THRESHOLD_GRID = np.round(np.concatenate([np.arange(0.80, 0.99, 0.005),
                                          np.arange(0.990, 0.9999, 0.0005)]), 5)


def threshold_table(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    y = labels != config.BENIGN_LABEL
    rows = []
    for t in THRESHOLD_GRID:
        pred = scores >= t
        m = binary_metrics(y, pred)
        row = {"threshold": float(t), **{k: v for k, v in m.items() if k != "confusion_matrix"},
               **m["confusion_matrix"]}
        for lab in config.ATTACK_LABELS:
            row[f"recall_{lab}"] = float(pred[labels == lab].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame, max_fpr: float) -> pd.Series:
    feasible = table[table.false_positive_rate <= max_fpr]
    if feasible.empty:
        return table.sort_values("threshold").iloc[-1]
    best = feasible.f1.max()
    return feasible[feasible.f1 >= best - 1e-12].sort_values("threshold").iloc[-1]


def calibrate(detector, calibration_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scores = detector.anomaly_score(calibration_df)
    labels = calibration_df[config.LABEL_COLUMN].to_numpy()
    table = threshold_table(scores, labels)
    points = {
        "strict": select_threshold(table, 0.01),
        "balanced": select_threshold(table, config.CALIBRATION_MAX_BENIGN_FPR),
        "sensitive": select_threshold(table, 0.05),
    }
    chosen = float(points["balanced"].threshold)
    detector.threshold = chosen
    detector.operating_points = {k: float(v.threshold) for k, v in points.items()}

    # Margin check: share of each class whose score lies within +/-0.005 of the
    # threshold. A large share means the class sits on a cliff and a small
    # distribution shift would flip it.
    margin = {lab: float(np.mean(np.abs(scores[labels == lab] - chosen) <= 0.005))
              for lab in (config.BENIGN_LABEL, *config.ATTACK_LABELS)}
    info = {
        "rule": f"max F1 s.t. benign FPR <= {config.CALIBRATION_MAX_BENIGN_FPR}",
        "n_calibration": int(len(labels)),
        "threshold": chosen,
        "operating_points": {k: v.to_dict() for k, v in points.items()},
        "share_within_0.005_of_threshold": margin,
        "median_score_by_class": {lab: float(np.median(scores[labels == lab]))
                                  for lab in (config.BENIGN_LABEL, *config.ATTACK_LABELS)},
    }
    detector.calibration_info = info
    return table, info
