"""Phase 7: evaluation on the labelled development partition.

Reports, for the frozen model and threshold:
  * holdout half (primary, unbiased: never used for any choice);
  * full eval partition (for comparability; half of it set the threshold);
  * holdout restricted to rows whose features do not occur in the training
    file (checks inflation from the 9,795 exact train/eval duplicates);
  * per-attack-class detection, confusion matrix, threshold comparison;
  * per-class score distribution plots.

Run:  python -m src.agents.anomaly_detection.evaluation
Writes outputs/anomaly_evaluation.json, outputs/threshold_comparison_holdout.csv,
       outputs/score_distribution.png
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config
from .calibrate import threshold_table
from .datasets import eval_split
from .detector import AnomalyDetector
from .metrics import binary_metrics, per_class_detection
from .preprocessing import load_flows

COMPARE_THRESHOLDS = (0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995)


def _report(scores, labels, threshold) -> dict:
    y = labels != config.BENIGN_LABEL
    pred = scores >= threshold
    return {
        "n": int(len(labels)),
        "threshold": threshold,
        **binary_metrics(y, pred),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "per_class": per_class_detection(labels, pred),
    }


def _plot(scores, labels, threshold, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, 1, 101)
    for lab in (config.BENIGN_LABEL, *config.ATTACK_LABELS):
        s = scores[labels == lab]
        axes[0].hist(s, bins=bins, histtype="step", density=True, label=f"{lab} (n={len(s)})")
        axes[1].hist(s, bins=np.linspace(0.9, 1, 101), histtype="step", density=True, label=lab)
    for ax in axes:
        ax.axvline(threshold, color="k", ls="--", lw=1, label=f"threshold {threshold:.3f}")
        ax.set_xlabel("anomaly_score (benign percentile)")
        ax.set_yscale("log")
    axes[0].set_title("Score distribution by class (holdout)")
    axes[1].set_title("Zoom: 0.90-1.00")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    det = AnomalyDetector.load(config.MODEL_PATH)
    t = det.threshold
    ev = load_flows(config.EVAL_PATH)
    cal, hold = eval_split(ev)

    s_hold = det.anomaly_score(hold)
    l_hold = hold[config.LABEL_COLUMN].to_numpy()
    s_full = det.anomaly_score(ev)
    l_full = ev[config.LABEL_COLUMN].to_numpy()

    # Rows of the holdout whose exact feature vector appears in the training file.
    train = load_flows(config.TRAIN_PATH)
    feats = [c for c in ev.columns if c not in config.METADATA_COLUMNS]
    train_hashes = set(pd.util.hash_pandas_object(train[feats], index=False))
    del train
    dup = pd.util.hash_pandas_object(hold[feats], index=False).isin(train_hashes).to_numpy()

    results = {
        "model_version": det.version,
        "threshold": t,
        "operating_points": det.operating_points,
        "score_semantics": "anomaly_score = fraction of held-out benign reference flows with a "
                           "lower-or-equal fused score; not an attack probability",
        "holdout": _report(s_hold, l_hold, t),
        "full_eval": _report(s_full, l_full, t),
        "holdout_excluding_train_duplicates": {
            "n_excluded": int(dup.sum()),
            **_report(s_hold[~dup], l_hold[~dup], t),
        },
        "threshold_comparison_holdout": [],
        "score_quantiles_by_class_holdout": {
            lab: {f"p{q}": float(np.percentile(s_hold[l_hold == lab], q)) for q in (1, 5, 25, 50, 75)}
            for lab in (config.BENIGN_LABEL, *config.ATTACK_LABELS)},
        "calibration": det.calibration_info,
    }
    table = threshold_table(s_hold, l_hold)
    table.to_csv(config.OUTPUT_DIR / "threshold_comparison_holdout.csv", index=False,
                 float_format="%.6f")
    for thr in COMPARE_THRESHOLDS:
        r = _report(s_hold, l_hold, thr)
        results["threshold_comparison_holdout"].append({
            "threshold": thr, **{k: r[k] for k in ("precision", "recall", "f1",
                                                   "false_positive_rate", "accuracy")},
            **{f"recall_{lab}": r["per_class"][lab]["flag_rate"] for lab in config.ATTACK_LABELS}})

    out = config.OUTPUT_DIR / "anomaly_evaluation.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    _plot(s_hold, l_hold, t, config.OUTPUT_DIR / "score_distribution.png")
    _print_summary(results)
    print("wrote", out)


def _print_summary(r: dict) -> None:
    for key in ("holdout", "full_eval", "holdout_excluding_train_duplicates"):
        m = r[key]
        cm = m["confusion_matrix"]
        print(f"\n[{key}] n={m['n']} threshold={m['threshold']:.3f}")
        print(f"  precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} "
              f"FPR={m['false_positive_rate']:.4f} accuracy={m['accuracy']:.4f} "
              f"ROC-AUC={m['roc_auc']:.4f} AP={m['average_precision']:.4f}")
        print(f"  confusion: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
        print("  per class: " + ", ".join(f"{k} {v['flagged']}/{v['n']} ({v['flag_rate']:.4f})"
                                         for k, v in m["per_class"].items()))
    print("\nthreshold comparison (holdout):")
    print(pd.DataFrame(r["threshold_comparison_holdout"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
