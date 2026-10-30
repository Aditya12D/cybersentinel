"""Read-only error analysis of the saved Agent 1 model on the eval set.
Never fits anything on eval data. Writes evaluation/error_analysis.json."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from agents.threat_detection import CLASSES, MODEL_DIR, ROOT, prepare

meta = json.loads((MODEL_DIR / "model_metadata.json").read_text())
feats = meta["features"]
model = joblib.load(MODEL_DIR / "threat_model.pkl")
tr = pd.read_parquet(ROOT / "data/Cyber/training/flows.parquet")
ev = pd.read_parquet(ROOT / "data/Cyber/eval/flows.parquet")
Xtr, Xev = prepare(tr, feats), prepare(ev, feats)
p = model.predict_proba(Xev)
pred = np.array(CLASSES)[p.argmax(1)]
wrong = np.where(pred != ev.label.to_numpy())[0]

# Per-class feature attribution (LightGBM SHAP-style contributions), shape (n, 5, nfeat+1)
contrib = model.booster_.predict(Xev.iloc[wrong], pred_contrib=True).reshape(len(wrong), 5, -1)

# Neighbourhood in a scale-free space: signed log1p, then standardise on TRAIN stats only
f = lambda X: np.sign(X) * np.log1p(np.abs(X.fillna(0).to_numpy()))
Ztr = f(Xtr); mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
nn = NearestNeighbors(n_neighbors=25).fit((Ztr - mu) / sd)

ps_med = Xtr[tr.label == "PORT_SCAN"].median()
bn_med = Xtr[tr.label == "BENIGN"].median()
out = []
for k, i in enumerate(wrong):
    true, guess = ev.label.iloc[i], pred[i]
    ti, gi = CLASSES.index(true), CLASSES.index(guess)
    push = pd.Series(contrib[k, gi, :-1] - contrib[k, ti, :-1], index=feats)  # >0 favours wrong class
    top = push.sort_values(ascending=False).head(8)
    dist, idx = nn.kneighbors(((f(Xev.iloc[[i]]) - mu) / sd))
    nb = tr.label.iloc[idx[0]]
    # how many training rows share this exact (Destination Port, packet-count) shape, by label
    same = tr[(Xtr["Destination Port"] == Xev["Destination Port"].iloc[i])
              & (Xtr["Total Fwd Packets"] == Xev["Total Fwd Packets"].iloc[i])
              & (Xtr["Total Backward Packets"] == Xev["Total Backward Packets"].iloc[i])]
    out.append({
        "record_id": ev.record_id.iloc[i], "true": true, "predicted": guess,
        "probabilities": dict(zip(CLASSES, np.round(p[i], 4).tolist())),
        "top_features_pushing_to_predicted": {
            c: {"push": round(float(v), 3), "value": float(Xev[c].iloc[i]),
                "PORT_SCAN_train_median": float(ps_med[c]), "BENIGN_train_median": float(bn_med[c])}
            for c, v in top.items()},
        "nearest_25_train_labels": nb.value_counts().to_dict(),
        "nearest_distance_min_median": [round(float(dist[0].min()), 3), round(float(np.median(dist[0])), 3)],
        "train_rows_same_port_and_packet_counts": same.label.value_counts().to_dict(),
    })

# Context: how often does the model emit a low-confidence prediction, and how accurate is it there?
conf = p.max(1)
bins = {f"<{t}": {"rows": int((conf < t).sum()),
                  "errors": int(((conf < t) & (pred != ev.label.to_numpy())).sum())}
        for t in (0.6, 0.8, 0.9, 0.99)}
res = {"errors": out, "eval_confidence_bands": bins}
(ROOT / "evaluation/error_analysis.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=1, default=str))
