"""Error analysis on the eval holdout half: false positives and missed attacks.

Questions answered with measurements (no tuning happens here):
  * Where do benign false positives come from (destination port is used for
    *analysis only*; it is not a model input)? Which detector member fires?
    How do false-positive HTTPS flows differ from correctly-passed HTTPS flows?
  * Would suppressing port 443 be justified? (Checked against what the data
    can and cannot show.)
  * Which attack flows are missed and why (slow DoS / brute force)?

Run:  python -m src.agents.anomaly_detection.error_analysis
Writes outputs/error_analysis.json
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from . import config
from .datasets import eval_split
from .detector import AnomalyDetector
from .evidence import compute_evidence
from .preprocessing import load_flows

PROFILE = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
           "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Flow Bytes/s",
           "Flow Packets/s", "Flow IAT Max", "Packet Length Mean", "Init_Win_bytes_forward",
           "Init_Win_bytes_backward", "FIN Flag Count", "RST Flag Count", "PSH Flag Count",
           "Active Mean", "Idle Mean"]


def _medians(df: pd.DataFrame) -> dict:
    return {c: float(df[c].median()) for c in PROFILE} if len(df) else {}


def _top_evidence(ev_lists, k=2, n=6):
    c = Counter(e["feature"] for ev in ev_lists for e in ev[:k])
    return c.most_common(n)


def _member(p_knn, p_if, t):
    return np.where((p_knn >= t) & (p_if >= t), "both",
                    np.where(p_knn >= t, "knn", np.where(p_if >= t, "iforest", "joint_only")))


def main() -> None:
    det = AnomalyDetector.load(config.MODEL_PATH)
    t = det.threshold
    _, hold = eval_split(load_flows(config.EVAL_PATH))
    hold = hold.reset_index(drop=True)
    res = det.score(hold)
    lab = hold[config.LABEL_COLUMN].to_numpy()
    port = hold["Destination Port"].to_numpy()
    flag = res.anomaly_score >= t
    benign = lab == config.BENIGN_LABEL
    fp, tn = benign & flag, benign & ~flag
    evidence = compute_evidence(det, res, flag)
    member = _member(res.knn_percentile, res.iforest_percentile, t)
    out: dict = {"threshold": t}

    # ---- false positives by port ------------------------------------------
    rows = []
    for p in pd.Series(port[benign]).value_counts().index[:12]:
        m = benign & (port == p)
        rows.append({"port": int(p), "benign_n": int(m.sum()),
                     "share_of_benign": float(m.sum() / benign.sum()),
                     "fp": int((m & flag).sum()), "fpr": float(flag[m].mean()),
                     "share_of_fp": float((m & flag).sum() / max(fp.sum(), 1))})
    out["false_positives_by_port"] = rows
    out["fp_member"] = dict(Counter(member[fp]))

    # ---- HTTPS: false positives vs correctly passed --------------------------
    h_fp, h_tn = fp & (port == 443), tn & (port == 443)
    out["https"] = {
        "n_fp": int(h_fp.sum()), "n_tn": int(h_tn.sum()),
        "fpr": float(h_fp.sum() / max((benign & (port == 443)).sum(), 1)),
        "member_of_fp": dict(Counter(member[h_fp])),
        "median_fp": _medians(hold[h_fp]), "median_tn": _medians(hold[h_tn]),
        "top_evidence_fp": _top_evidence([evidence[i] for i in np.where(h_fp)[0]]),
        "fp_severity_counts": dict(Counter(e["severity"] for i in np.where(h_fp)[0]
                                           for e in evidence[i])),
        "fp_flow_duration_ge_100s": float((hold.loc[h_fp, "Flow Duration"] >= 1e8).mean()),
        "tn_flow_duration_ge_100s": float((hold.loc[h_tn, "Flow Duration"] >= 1e8).mean()),
    }
    attack_ports = {lab_: sorted(pd.Series(port[lab == lab_]).value_counts().index[:3].astype(int).tolist())
                    for lab_ in config.ATTACK_LABELS}
    out["port_443_suppression_check"] = {
        "attack_flows_on_443_in_holdout": int(((~benign) & (port == 443)).sum()),
        "top_ports_per_attack_class": attack_ports,
        "fp_removed_if_443_suppressed": int(h_fp.sum()),
        "conclusion": ("The eval data contains (almost) no attacks on port 443, so suppressing 443 "
                       "would look free on this data while blinding the agent to any HTTPS-borne "
                       "attack in the replay; the data provides no evidence that 443 traffic is "
                       "safe, so no port-based suppression is applied."),
    }

    # ---- missed attacks ----------------------------------------------------
    miss = {}
    for cls in config.ATTACK_LABELS:
        m_miss = (lab == cls) & ~flag
        m_det = (lab == cls) & flag
        if not m_miss.any():
            miss[cls] = {"n_missed": 0}
            continue
        d = hold[m_miss]
        miss[cls] = {
            "n_missed": int(m_miss.sum()), "n": int((lab == cls).sum()),
            "median_missed": _medians(d), "median_detected": _medians(hold[m_det]),
            "missed_score_quantiles": {f"p{q}": float(np.percentile(res.anomaly_score[m_miss], q))
                                       for q in (5, 50, 95)},
            "missed_knn_pct_median": float(np.median(res.knn_percentile[m_miss])),
            "missed_iforest_pct_median": float(np.median(res.iforest_percentile[m_miss])),
            "share_duration_gt_5s": float((d["Flow Duration"] > 5e6).mean()),
            "share_packet_rate_lt_2": float((d["Flow Packets/s"] < 2).mean()),
            "share_rst_zero": float((d["RST Flag Count"] == 0).mean()),
            "detected_share_duration_gt_5s": float((hold.loc[m_det, "Flow Duration"] > 5e6).mean()),
        }
    out["missed_attacks"] = miss
    benign_slow = benign & (hold["Flow Duration"] > 5e6).to_numpy() & (hold["Flow Packets/s"] < 2).to_numpy()
    out["benign_slow_long_flows"] = {"n": int(benign_slow.sum()),
                                     "share_of_benign": float(benign_slow.mean() / benign.mean()),
                                     "fpr_within": float(flag[benign_slow].mean())}

    path = config.OUTPUT_DIR / "error_analysis.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=lambda v: round(float(v), 4))[:9000])
    print("wrote", path)


if __name__ == "__main__":
    main()
