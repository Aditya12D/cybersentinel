"""Ablation: kNN-only vs Isolation-Forest-only vs fused, on identical splits.

For every variant the score is a benign percentile computed on the same
held-out benign reference set, the threshold is chosen with the *same* rule on
the eval calibration half (max F1 s.t. benign FPR <= 3%, see calibrate.py), and
metrics are reported once on the untouched holdout half. Scoring latency of
each member is measured on the holdout as well.

Also evaluated (cheap, targets false positives): kNN with *all* benign fit
flows as prototypes instead of the 100k sample.

Run:  python -m src.agents.anomaly_detection.ablation      (~2 min)
Writes outputs/ablation.json
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from . import config
from .calibrate import select_threshold, threshold_table
from .datasets import load_splits
from .detector import AnomalyDetector, _percentile
from .metrics import binary_metrics, per_class_detection


def _evaluate(name, ref_raw, cal_raw, hold_raw, lab_cal, lab_hold) -> dict:
    """Map raw member scores to benign percentiles, calibrate, evaluate."""
    srt = np.sort(ref_raw)
    s_cal, s_hold = _percentile(srt, cal_raw), _percentile(srt, hold_raw)
    table = threshold_table(s_cal, lab_cal)
    t = float(select_threshold(table, config.CALIBRATION_MAX_BENIGN_FPR).threshold)
    y = lab_hold != config.BENIGN_LABEL
    pred = s_hold >= t
    m = binary_metrics(y, pred)
    out = {"threshold": t, **{k: v for k, v in m.items()}, "roc_auc": float(roc_auc_score(y, s_hold)),
           "pr_auc": float(average_precision_score(y, s_hold)),
           "per_class": {k: v["flag_rate"] for k, v in per_class_detection(lab_hold, pred).items()}}
    print(f"{name:34s} t={t:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
          f"FPR={m['false_positive_rate']:.4f} AUC={out['roc_auc']:.4f} PR-AUC={out['pr_auc']:.4f} "
          + " ".join(f"{k[:5]}={v:.3f}" for k, v in out["per_class"].items() if k != "BENIGN"))
    return out


def _timed(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return r, time.perf_counter() - t0


def main() -> None:
    det = AnomalyDetector.load(config.MODEL_PATH)
    sp = load_splits()
    lab_cal = sp.eval_calibration[config.LABEL_COLUMN].to_numpy()
    lab_hold = sp.eval_holdout[config.LABEL_COLUMN].to_numpy()
    Z = {k: det.preprocessor.transform(d)[:, det.model_idx_] for k, d in
         (("ref", sp.benign_reference), ("cal", sp.eval_calibration), ("hold", sp.eval_holdout))}

    knn = {k: det._knn(v)[0] for k, v in Z.items()}
    iff = {k: det._iforest_score(v) for k, v in Z.items()}
    results = {}
    results["knn_only"] = _evaluate("kNN only (k=5, 100k protos)", knn["ref"], knn["cal"], knn["hold"],
                                    lab_cal, lab_hold)
    results["iforest_only"] = _evaluate("Isolation Forest only", iff["ref"], iff["cal"], iff["hold"],
                                        lab_cal, lab_hold)

    def fused(k):
        return 0.5 * (_percentile(det.ref_knn_, knn[k]) + _percentile(det.ref_if_, iff[k]))
    results["fused_mean_percentile"] = _evaluate("fused (deployed)", fused("ref"), fused("cal"),
                                                 fused("hold"), lab_cal, lab_hold)

    # Variant: every benign fit flow as a prototype.
    Zfit = det.preprocessor.transform(sp.benign_fit)[:, det.model_idx_]
    nn_all = NearestNeighbors(n_neighbors=config.KNN_N_NEIGHBORS).fit(Zfit)
    knn_all = {k: nn_all.kneighbors(v)[0].mean(axis=1) for k, v in Z.items()}
    results["knn_only_all_prototypes"] = _evaluate(
        f"kNN only ({len(Zfit)} protos)", knn_all["ref"], knn_all["cal"], knn_all["hold"],
        lab_cal, lab_hold)
    sa = {k: 0.5 * (_percentile(np.sort(knn_all["ref"]), knn_all[k]) + _percentile(det.ref_if_, iff[k]))
          for k in Z}
    results["fused_all_prototypes"] = _evaluate("fused, all protos", sa["ref"], sa["cal"], sa["hold"],
                                                lab_cal, lab_hold)

    # Latency per member on the holdout (batch) and for single rows.
    X = Z["hold"]
    _, t_knn = _timed(det._knn, X)
    _, t_if = _timed(det.iforest_.score_samples, X)
    _, t_knn_all = _timed(lambda a: nn_all.kneighbors(a), X)
    one = X[:1]
    reps = 200
    _, t1_knn = _timed(lambda: [det._knn(one) for _ in range(reps)])
    _, t1_if = _timed(lambda: [det._iforest_score(one) for _ in range(reps)])
    _, t1_knn_all = _timed(lambda: [nn_all.kneighbors(one) for _ in range(reps)])
    results["latency_ms"] = {
        "batch_per_1k_rows": {"knn_100k": 1e3 * t_knn / len(X) * 1e3,
                              "knn_all": 1e3 * t_knn_all / len(X) * 1e3,
                              "iforest": 1e3 * t_if / len(X) * 1e3},
        "single_row": {"knn_100k": 1e3 * t1_knn / reps, "knn_all": 1e3 * t1_knn_all / reps,
                       "iforest_vectorised": 1e3 * t1_if / reps},
    }
    print("latency ms:", json.dumps(results["latency_ms"], default=lambda v: round(v, 3)))
    out = config.OUTPUT_DIR / "ablation.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
