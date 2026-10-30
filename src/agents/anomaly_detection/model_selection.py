"""Phase 2: reproducible comparison of candidate anomaly detectors.

Every candidate is fitted on BENIGN training flows only (``benign_fit``).
Comparison data:
  * ROC-AUC / average precision on the eval *calibration* half (holdout untouched);
  * label-free operating points: thresholds at the 99/98/97/95th percentile of
    scores on the held-out BENIGN reference set (target benign FPR 1/2/3/5%),
    with resulting recall per attack class on the calibration half.

Candidates: Isolation Forest (several max_samples), One-Class SVM (exact RBF on
a 20k subsample, and Nystroem + SGD on all data), statistical baselines (robust
z RMS / max, Ledoit-Wolf Mahalanobis), kNN distance, label-free correlation
pruning, rank-averaged ensembles and a Destination Port ablation.

Run:  python -m src.agents.anomaly_detection.model_selection   (~3 min)
Writes outputs/model_selection.json
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

from . import config
from .datasets import load_splits
from .detector import _percentile, correlation_prune
from .preprocessing import FlowPreprocessor, binary_target

SEED = config.RANDOM_SEED
FPR_TARGETS = (0.01, 0.02, 0.03, 0.05)


def _sub(X, n):
    rng = np.random.default_rng(SEED)
    return X if len(X) <= n else X[rng.choice(len(X), n, replace=False)]


def iforest(max_samples):
    def fit(X):
        m = IsolationForest(n_estimators=300, max_samples=max_samples, random_state=SEED,
                            n_jobs=-1).fit(X)
        return lambda Z: -m.score_samples(Z)
    return fit


def ocsvm_exact(X):
    m = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale").fit(_sub(X, 20_000))
    return lambda Z: -m.decision_function(Z)


def ocsvm_nystroem(X):
    gamma = 1.0 / (X.shape[1] * X.var())
    fm = Nystroem(gamma=gamma, n_components=500, random_state=SEED).fit(_sub(X, 20_000))
    m = SGDOneClassSVM(nu=0.01, random_state=SEED, max_iter=20, tol=None).fit(fm.transform(X))
    return lambda Z: -m.decision_function(fm.transform(Z))


def robust_z_rms(X):
    return lambda Z: np.sqrt(np.mean(Z.astype(np.float64) ** 2, axis=1))


def robust_z_max(X):
    return lambda Z: np.abs(Z).max(axis=1)


def mahalanobis(X):
    lw = LedoitWolf().fit(_sub(X, 50_000))
    return lambda Z: lw.mahalanobis(Z.astype(np.float64))


def knn(k, n_proto):
    def fit(X):
        nn = NearestNeighbors(n_neighbors=k).fit(_sub(X, n_proto))
        return lambda Z: nn.kneighbors(Z)[0].mean(axis=1)
    return fit


def summarise(s_ref, s_cal, y, labels) -> dict:
    res = {"roc_auc": round(float(roc_auc_score(y, s_cal)), 4),
           "average_precision": round(float(average_precision_score(y, s_cal)), 4)}
    for target in FPR_TARGETS:
        thr = np.quantile(s_ref, 1 - target)
        pred = s_cal > thr
        point = {"benign_fpr": round(float(pred[y == 0].mean()), 4),
                 "attack_recall": round(float(pred[y == 1].mean()), 4)}
        for lab in config.ATTACK_LABELS:
            point[f"recall_{lab}"] = round(float(pred[labels == lab].mean()), 4)
        res[f"at_reference_fpr_{int(target * 100)}pct"] = point
    return res


def main():
    sp = load_splits()
    labels = sp.eval_calibration[config.LABEL_COLUMN].to_numpy()
    y = binary_target(sp.eval_calibration[config.LABEL_COLUMN])

    pre = FlowPreprocessor().fit(sp.benign_fit)
    Xf, Xr, Xc = (pre.transform(d) for d in (sp.benign_fit, sp.benign_reference,
                                               sp.eval_calibration))
    keep = correlation_prune(Xf, config.MODEL_FEATURE_CORRELATION_PRUNE, SEED)
    print(f"{Xf.shape[1]} features after preprocessing, {len(keep)} after correlation pruning")

    candidates = {
        "isolation_forest_ms256": iforest(256),
        "isolation_forest_ms1024": iforest(1024),
        "isolation_forest_ms4096": iforest(4096),
        "ocsvm_rbf_exact_20k": ocsvm_exact,
        "ocsvm_nystroem_sgd_full": ocsvm_nystroem,
        "stat_robust_z_rms": robust_z_rms,
        "stat_robust_z_max": robust_z_max,
        "stat_mahalanobis_ledoitwolf": mahalanobis,
        "knn5_50k": knn(5, 50_000),
        "knn5_100k": knn(5, 100_000),
        "knn1_100k": knn(1, 100_000),
    }
    results, pct = {}, {}
    for variant, cols in (("all", None), ("pruned", keep)):
        A, B, C = (Xf, Xr, Xc) if cols is None else (Xf[:, cols], Xr[:, cols], Xc[:, cols])
        for name, fit in candidates.items():
            if variant == "all" and name.startswith("knn") and name != "knn5_100k":
                continue
            key = f"{name}__{variant}"
            t0 = time.perf_counter()
            scorer = fit(A)
            t_fit = time.perf_counter() - t0
            t0 = time.perf_counter()
            s_cal = scorer(C)
            t_score = time.perf_counter() - t0
            s_ref = scorer(B)
            results[key] = {"fit_seconds": round(t_fit, 2),
                            "score_ms_per_1k": round(1e6 * t_score / len(C), 1),
                            **summarise(s_ref, s_cal, y, labels)}
            srt = np.sort(s_ref)
            pct[key] = (_percentile(srt, s_ref), _percentile(srt, s_cal))
            print(key, json.dumps(results[key]))

    # Rank-averaged (benign percentile) ensembles of complementary members.
    for a, b in (("knn5_100k", "isolation_forest_ms4096"), ("knn5_100k", "stat_mahalanobis_ledoitwolf"),
                 ("knn5_100k", "ocsvm_rbf_exact_20k")):
        ka, kb = f"{a}__pruned", f"{b}__pruned"
        for how, op in (("mean", lambda u, v: 0.5 * (u + v)), ("max", np.maximum)):
            key = f"ensemble_{how}({a},{b})__pruned"
            results[key] = summarise(op(pct[ka][0], pct[kb][0]), op(pct[ka][1], pct[kb][1]),
                                     y, labels)
            print(key, json.dumps(results[key]))

    # Ablation: keep Destination Port as a numeric feature.
    saved = dict(config.FORCED_EXCLUSIONS)
    config.FORCED_EXCLUSIONS.clear()
    try:
        pre_p = FlowPreprocessor().fit(sp.benign_fit)
        A, B, C = (pre_p.transform(d) for d in (sp.benign_fit, sp.benign_reference,
                                                 sp.eval_calibration))
        sc = iforest(4096)(A)
        results["ablation_isolation_forest_ms4096_with_port__all"] = summarise(sc(B), sc(C), y, labels)
        print("ablation_with_port", json.dumps(results["ablation_isolation_forest_ms4096_with_port__all"]))
    finally:
        config.FORCED_EXCLUSIONS.update(saved)

    out = config.OUTPUT_DIR / "model_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"preprocessing": pre.describe(),
                               "pruned_model_features": [pre.features_[i] for i in keep],
                               "candidates": results}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
