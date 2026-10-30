"""Train + calibrate Agent 2 and save the model bundle.

Run:  python -m src.agents.anomaly_detection.train
Outputs:
    models/anomaly_detector.joblib      (git-ignored)
    outputs/threshold_analysis.csv      (calibration-half threshold sweep)
    outputs/calibration.json
"""
from __future__ import annotations

import json
import time

from . import config
from .calibrate import calibrate
from .datasets import load_splits
from .detector import AnomalyDetector


def main() -> None:
    t0 = time.perf_counter()
    sp = load_splits()
    print(f"benign fit={len(sp.benign_fit)} reference={len(sp.benign_reference)} "
          f"eval calibration={len(sp.eval_calibration)} holdout={len(sp.eval_holdout)}")

    det = AnomalyDetector().fit(sp.benign_fit, sp.benign_reference)
    print(f"fitted in {time.perf_counter() - t0:.1f}s; "
          f"{len(det.features)} preprocessed features, {len(det.model_features)} model features")

    table, info = calibrate(det, sp.eval_calibration)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.OUTPUT_DIR / "threshold_analysis.csv", index=False, float_format="%.6f")
    (config.OUTPUT_DIR / "calibration.json").write_text(json.dumps(
        {"preprocessing": det.preprocessor.describe(), "model_features": det.model_features,
         "calibration": info}, indent=2, default=float))
    det.save(config.MODEL_PATH)
    bp = info["operating_points"]["balanced"]
    print(f"threshold={det.threshold:.4f}  cal FPR={bp['false_positive_rate']:.4f} "
          f"recall={bp['recall']:.4f} precision={bp['precision']:.4f} F1={bp['f1']:.4f}")
    print("operating points:", det.operating_points)
    print(f"saved {config.MODEL_PATH} ({config.MODEL_PATH.stat().st_size / 1e6:.1f} MB) "
          f"in {time.perf_counter() - t0:.1f}s total")


if __name__ == "__main__":
    main()
