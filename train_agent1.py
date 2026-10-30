"""Agent 1 training + benchmarking.

  python train_agent1.py bench                 # compare models -> evaluation/model_comparison.csv
  python train_agent1.py weights --model lgbm  # class-weighting study on one model
  python train_agent1.py final --model lgbm --power 0.5   # fit + save artifacts

Model selection uses a stratified 15% validation split of TRAINING data only.
The official eval set is reported, never fitted on. "eval_unseen" excludes the
9,795 eval rows whose features exactly match a training row.
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score)
from sklearn.model_selection import train_test_split

from agents.threat_detection import CLASSES, DROP, MODEL_DIR, ROOT, feature_columns, prepare

SEED = 42
DATA = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "evaluation"


def load():
    tr = pd.read_parquet(DATA / "training" / "flows.parquet")
    ev = pd.read_parquet(DATA / "eval" / "flows.parquet")
    feats = feature_columns(tr.columns)
    enc = {c: i for i, c in enumerate(CLASSES)}
    Xtr, ytr = prepare(tr, feats), tr["label"].map(enc).to_numpy()
    Xev, yev = prepare(ev, feats), ev["label"].map(enc).to_numpy()
    h = lambda X: pd.util.hash_pandas_object(X, index=False)
    unseen = ~h(Xev).isin(set(h(Xtr))).to_numpy()
    return feats, Xtr, ytr, Xev, yev, unseen


def class_weights(y, power):
    """power 0 = none, 0.5 = sqrt-balanced, 1 = fully balanced."""
    counts = np.bincount(y, minlength=len(CLASSES))
    w = (len(y) / (len(CLASSES) * counts)) ** power
    return w[y]


def make(name, n_estimators=None):
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="multiclass", n_estimators=n_estimators or 2000,
                                  learning_rate=0.05, num_leaves=63, min_child_samples=20,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(objective="multi:softprob", n_estimators=n_estimators or 2000,
                                 learning_rate=0.05, max_depth=10, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                                 early_stopping_rounds=None if n_estimators else 100,
                                 eval_metric="mlogloss", random_state=SEED, n_jobs=-1)
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(loss_function="MultiClass", iterations=n_estimators or 3000,
                                  learning_rate=0.1, depth=8, random_seed=SEED,
                                  thread_count=-1, verbose=0)
    if name == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
    raise ValueError(name)


def fit(name, X, y, Xv=None, yv=None, power=0.0, n_estimators=None):
    """Returns (model, best_iteration or None). Early-stops on (Xv, yv) when given."""
    m, sw = make(name, n_estimators), class_weights(y, power)
    if name == "lgbm" and Xv is not None:
        import lightgbm as lgb
        m.fit(X, y, sample_weight=sw, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        return m, m.best_iteration_
    if name == "xgb" and Xv is not None:
        m.fit(X, y, sample_weight=sw, eval_set=[(Xv, yv)], verbose=False)
        return m, m.best_iteration + 1
    if name == "cat" and Xv is not None:
        m.fit(X, y, sample_weight=sw, eval_set=(Xv, yv), early_stopping_rounds=100)
        return m, m.get_best_iteration() + 1
    m.fit(X, y, sample_weight=sw)
    return m, None


def metrics(y, p):
    pred = p.argmax(1)
    pc = f1_score(y, pred, average=None, labels=range(len(CLASSES)))
    rc = recall_score(y, pred, average=None, labels=range(len(CLASSES)))
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "macro_precision": precision_score(y, pred, average="macro"),
        "macro_recall": recall_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "log_loss": log_loss(y, np.clip(p, 1e-15, 1), labels=range(len(CLASSES))),
        "ece": ece(y, p),
        **{f"f1_{c}": v for c, v in zip(CLASSES, pc)},
        **{f"recall_{c}": v for c, v in zip(CLASSES, rc)},
    }


def ece(y, p, bins=15):
    """Expected calibration error of the max-probability confidence."""
    conf, pred = p.max(1), p.argmax(1)
    edges, err = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return err


def split(Xtr, ytr):
    return train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=SEED)


def row(name, power, m, best, secs, Xa, ya, Xev, yev, unseen):
    t = time.perf_counter(); pe = m.predict_proba(Xev); inf = time.perf_counter() - t
    va, ev, un = metrics(ya, m.predict_proba(Xa)), metrics(yev, pe), metrics(yev[unseen], pe[unseen])
    r = {"model": name, "power": power, "best_iter": best, "train_s": round(secs, 1),
         "infer_us_per_row": round(inf / len(Xev) * 1e6, 2),
         **{f"val_{k}": v for k, v in va.items()},
         **{f"eval_{k}": v for k, v in ev.items()},
         **{f"unseen_{k}": v for k, v in un.items()}}
    print(f"{name:5} p={power:<4} val_mF1={va['macro_f1']:.5f} eval_mF1={ev['macro_f1']:.5f} "
          f"unseen_mF1={un['macro_f1']:.5f} acc={ev['accuracy']:.5f} "
          f"BF_rec={ev['recall_BRUTE_FORCE']:.4f} BF_f1={ev['f1_BRUTE_FORCE']:.4f} "
          f"ece={ev['ece']:.4f} t={secs:.0f}s", flush=True)
    return r


def run(names, powers, out):
    feats, Xtr, ytr, Xev, yev, unseen = load()
    Xa, Xb, ya, yb = split(Xtr, ytr)
    rows = []
    for name in names:
        for power in powers:
            try:
                t = time.perf_counter(); m, best = fit(name, Xa, ya, Xb, yb, power)
                rows.append(row(name, power, m, best, time.perf_counter() - t, Xb, yb, Xev, yev, unseen))
            except ImportError as e:
                print(f"skip {name}: {e}")
    EVAL_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(EVAL_DIR / out, index=False)


def final(name, power):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    feats, Xtr, ytr, Xev, yev, unseen = load()
    Xa, Xb, ya, yb = split(Xtr, ytr)
    _, best = fit(name, Xa, ya, Xb, yb, power)           # pick iteration count on val
    t = time.perf_counter()
    m, _ = fit(name, Xtr, ytr, power=power, n_estimators=best)  # refit on all training rows
    secs = time.perf_counter() - t
    MODEL_DIR.mkdir(exist_ok=True); EVAL_DIR.mkdir(exist_ok=True)
    joblib.dump(m, MODEL_DIR / "threat_model.pkl")
    pe = m.predict_proba(Xev); pred = pe.argmax(1)
    ev, un = metrics(yev, pe), metrics(yev[unseen], pe[unseen])
    meta = {"model": name, "params": {k: str(v) for k, v in m.get_params().items()},
            "class_weight_power": power, "n_estimators": best, "classes": CLASSES,
            "features": feats, "dropped_features": DROP, "seed": SEED,
            "train_rows": int(len(ytr)), "train_seconds": round(secs, 1),
            "preprocessing": "reindex to features, float32, +/-inf -> NaN, NaN left to model"}
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    (EVAL_DIR / "agent1_metrics.json").write_text(json.dumps(
        {"eval": ev, "eval_unseen_rows": un, "n_eval": int(len(yev)),
         "n_eval_unseen": int(unseen.sum())}, indent=2))
    (EVAL_DIR / "classification_report.json").write_text(json.dumps(
        classification_report(yev, pred, target_names=CLASSES, output_dict=True, digits=5), indent=2))
    cm = confusion_matrix(yev, pred)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(EVAL_DIR / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(np.log1p(cm), cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
    ax.set_xticks(range(5), CLASSES, rotation=30); ax.set_yticks(range(5), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Agent 1 ({name}) on eval  macro-F1={ev['macro_f1']:.4f}")
    fig.tight_layout(); fig.savefig(EVAL_DIR / "confusion_matrix.png", dpi=130)
    if hasattr(m, "feature_importances_"):
        pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False) \
          .to_csv(EVAL_DIR / "feature_importance.csv", header=["importance"])
    print(json.dumps({k: round(v, 5) for k, v in ev.items()}, indent=1))
    print("unseen macro_f1", round(un["macro_f1"], 5)); print(cm)


def cv(k=5):
    """Stratified k-fold on DE-DUPLICATED training rows (identical flows can't straddle
    folds). Fixed tree counts from the final fit, so no early stopping on the test fold.
    Compares LightGBM, XGBoost and their probability average on the same folds."""
    from sklearn.model_selection import StratifiedKFold
    feats, Xtr, ytr, *_ = load()
    keep = ~Xtr.duplicated().to_numpy()
    X, y = Xtr[keep].reset_index(drop=True), ytr[keep]
    print(f"cv on {len(y)} unique rows ({(~keep).sum()} duplicates removed)", flush=True)
    rows = []
    for f, (a, b) in enumerate(StratifiedKFold(k, shuffle=True, random_state=SEED).split(X, y)):
        p = {n: fit(n, X.iloc[a], y[a], n_estimators=it)[0].predict_proba(X.iloc[b])
             for n, it in [("lgbm", 1999), ("xgb", 1136)]}
        p["lgbm+xgb"] = (p["lgbm"] + p["xgb"]) / 2
        for n, pr in p.items():
            r = metrics(y[b], pr)
            rows.append({"fold": f, "model": n, "errors": int((pr.argmax(1) != y[b]).sum()), **r})
            print(f"fold {f} {n:9} errors={rows[-1]['errors']:3} macro_f1={r['macro_f1']:.5f} "
                  f"BF_rec={r['recall_BRUTE_FORCE']:.4f}", flush=True)
    d = pd.DataFrame(rows); d.to_csv(EVAL_DIR / "cv_results.csv", index=False)
    print(d.groupby("model")[["errors", "macro_f1", "recall_BRUTE_FORCE", "log_loss"]]
          .agg(["mean", "std"]).round(6).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "weights", "final", "cv"])
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--power", type=float, default=0.0)
    ap.add_argument("--models", default="et,rf,lgbm,xgb,cat")
    ap.add_argument("--out", default="model_comparison.csv")
    a = ap.parse_args()
    if a.cmd == "bench":
        run(a.models.split(","), [0.0], a.out)
    elif a.cmd == "weights":
        run([a.model], [0.0, 0.5, 1.0], f"weights_{a.model}.csv")
    elif a.cmd == "cv":
        cv()
    else:
        final(a.model, a.power)
