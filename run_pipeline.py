"""CyberSentinel end-to-end pipeline: Agent 1 + Agent 2 (independently) -> Agent 3.

  python run_pipeline.py                                   # official eval set + report
  python run_pipeline.py --input path/flows.parquet        # any flows file (labels optional)
  python run_pipeline.py --bayes                           # EXPERIMENTAL escalation rule on

Each model loads once (both agents cache it). If an agent's batch call raises,
that agent is re-run per flow so one bad flow cannot take the batch down; the
failing flows reach Agent 3 as exceptions and are BLOCKed as INVALID_EVIDENCE.
Labels, when present, are used only for the report, never by any agent.

Writes outputs/pipeline_report.json (metrics), outputs/pipeline_predictions.csv
and outputs/pipeline_results.jsonl (full per-flow results with rationale).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

try:
    import resource  # POSIX only; absent on Windows
except ImportError:
    resource = None

from agents.threat_detection import predict_batch, predict_threat
from src.agents.anomaly_detection import detect_anomalies, detect_anomaly
from src.agents.evidence_fusion import (ALLOW, BLOCK, CLASSES, DEFAULT_CONFIG, UNKNOWN,
                                        FusionConfig, fuse, fuse_batch)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
log = logging.getLogger("cybersentinel.pipeline")


def _run_agent(fn, df: pd.DataFrame, name: str) -> tuple[list, float]:
    t = time.perf_counter()
    try:
        out = fn(df)
        if len(out) != len(df):
            raise RuntimeError(f"{name} returned {len(out)} results for {len(df)} flows")
    except Exception as e:
        # ponytail: per-flow retry is ~4 ms/flow; bisect chunks if whole-set retries get slow
        log.warning("pipeline.batch_failed agent=%s error=%r; retrying per flow", name, e)
        out = []
        for i in range(len(df)):
            try:
                out.append(fn(df.iloc[[i]])[0])
            except Exception as e:  # handed to Agent 3 as evidence of failure, not swallowed
                log.error("pipeline.flow_failed agent=%s row=%d error=%r", name, i, e)
                out.append(e)
    return out, time.perf_counter() - t


def run(df: pd.DataFrame, cfg: FusionConfig = DEFAULT_CONFIG):
    """Returns (agent1 outputs, agent2 outputs, fused results, seconds per stage)."""
    a1, t1 = _run_agent(predict_batch, df, "threat_detection")
    a2, t2 = _run_agent(detect_anomalies, df, "anomaly_detection")
    ids = df["record_id"].astype(str).tolist() if "record_id" in df else None
    t = time.perf_counter()
    fused = fuse_batch(a1, a2, ids, cfg)
    return a1, a2, fused, {"agent1_s": t1, "agent2_s": t2, "agent3_s": time.perf_counter() - t}


def _class_metrics(y, pred) -> dict:
    rep = classification_report(y, pred, labels=list(CLASSES), output_dict=True, zero_division=0)
    return {"n": int(len(y)), "accuracy": float(np.mean(y == pred)),
            "macro_f1": float(f1_score(y, pred, labels=list(CLASSES), average="macro", zero_division=0)),
            "errors": int(np.sum(y != pred)),
            "per_class": {c: {k: round(rep[c][k], 6) for k in ("precision", "recall", "f1-score", "support")}
                          for c in CLASSES}}


def report(df, a1, a2, fused, times, cfg) -> dict:
    y = df["label"].astype(str).to_numpy()
    attack = y != "BENIGN"
    p1 = np.array([r["predicted_class"] if isinstance(r, dict) else "ERROR" for r in a1])
    ok2 = np.array([isinstance(r, dict) for r in a2])
    f2 = np.array([bool(r["is_anomaly"]) if isinstance(r, dict) else False for r in a2])
    fin = np.array([r["final_class"] for r in fused])
    dec = np.array([r["decision"] for r in fused])
    rule = np.array([r["rule"] for r in fused])
    tp, fp = int(np.sum(f2 & attack & ok2)), int(np.sum(f2 & ~attack & ok2))
    fn_, tn = int(np.sum(~f2 & attack & ok2)), int(np.sum(~f2 & ~attack & ok2))
    false_allow, false_block = attack & (dec == ALLOW), ~attack & (dec == BLOCK)
    examples = {}
    for r, lab in zip(fused, y):
        if len(examples.setdefault(r["rule"], [])) < 2:
            examples[r["rule"]].append({"true_label": lab, **{k: r[k] for k in (
                "record_id", "final_class", "decision", "evidence_status", "rationale")}})
    n = len(df)
    return {
        "input_rows": n,
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "agent1": _class_metrics(y, p1),
        "agent2": {
            "note": ("anomaly_score is a benign-unusualness percentile, not an attack probability. "
                     "Agent 2's threshold (0.97) was originally selected on half of this eval set; it "
                     "equals a 3% benign false-positive budget by construction, confirmed training-only."),
            "threshold": a2[0]["threshold"] if ok2.any() else None,
            "valid_outputs": int(ok2.sum()),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn_, 1),
            "false_positive_rate": fp / max(fp + tn, 1), "false_negative_rate": fn_ / max(fn_ + tp, 1),
            "confusion": {"tn": tn, "fp": fp, "fn": fn_, "tp": tp},
            "detection_rate_by_class": {c: float(f2[y == c].mean()) for c in CLASSES if (y == c).any()},
        },
        "agent3": {
            **_class_metrics(y, fin),
            "confusion_matrix": {"labels_true": list(CLASSES), "labels_pred": [*CLASSES, UNKNOWN],
                                 "matrix": confusion_matrix(y, fin, labels=[*CLASSES, UNKNOWN])[:5].tolist()},
            "decisions": dict(Counter(dec.tolist())),
            "evidence_status": dict(Counter(r["evidence_status"] for r in fused)),
            "rules": dict(Counter(rule.tolist())),
            "needs_review": int(sum(r["needs_review"] for r in fused)),
            "false_allow": {"count": int(false_allow.sum()), "by_rule": dict(Counter(rule[false_allow].tolist())),
                            "record_ids": df["record_id"].astype(str).to_numpy()[false_allow].tolist()[:20]},
            "false_block": {"count": int(false_block.sum()), "by_rule": dict(Counter(rule[false_block].tolist()))},
            "decision_accuracy": float(np.mean((dec == BLOCK) == attack)),
        },
        "examples": examples,
        "timing": {**{k: round(v, 3) for k, v in times.items()},
                   "total_ms_per_flow": round(sum(times.values()) / n * 1e3, 4)},
    }


def single_flow_latency(df: pd.DataFrame, k: int = 100) -> dict:
    ms = []
    for _, row in df.head(k).iterrows():
        rec = row.to_dict()
        t = time.perf_counter()
        fuse(predict_threat(rec), detect_anomaly(rec), record_id=str(rec.get("record_id")))
        ms.append((time.perf_counter() - t) * 1e3)
    return {"n": len(ms), "p50_ms": round(float(np.percentile(ms, 50)), 3),
            "p99_ms": round(float(np.percentile(ms, 99)), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data" / "raw" / "eval" / "flows.parquet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bayes", action="store_true", help="enable EXPERIMENTAL Bayes escalation")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("src.agents.evidence_fusion").setLevel(logging.ERROR)  # per-flow noise; counted in report

    cfg = FusionConfig(bayes_escalation=a.bayes)
    df = pd.read_parquet(a.input)
    if a.limit:
        df = df.head(a.limit)
    log.info("pipeline.start rows=%d input=%s bayes=%s", len(df), a.input, a.bayes)
    predict_batch(df.head(1)), detect_anomalies(df.head(1))  # load both models once, outside timings
    a1, a2, fused, times = run(df, cfg)
    again = fuse_batch(a1, a2, df["record_id"].astype(str).tolist() if "record_id" in df else None, cfg)
    deterministic = again == fused

    OUT.mkdir(exist_ok=True)
    pd.DataFrame([{k: r[k] for k in ("record_id", "final_class", "decision", "evidence_status",
                                     "needs_review", "rule")} | r["signals"] for r in fused]) \
      .drop(columns="anomaly_features").to_csv(OUT / "pipeline_predictions.csv", index=False)
    with open(OUT / "pipeline_results.jsonl", "w") as f:
        for r in fused:
            f.write(json.dumps(r) + "\n")

    rep = {"timing_note": "seconds per stage after model load"}
    if "label" in df:
        rep = report(df, a1, a2, fused, times, cfg)
        rep["single_flow_latency"] = single_flow_latency(df)
    rep["deterministic_rerun"] = deterministic
    if resource is not None:
        rep["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                   / (1 << 20 if sys.platform == "darwin" else 1 << 10), 1)
    name = "pipeline_report_bayes.json" if a.bayes else "pipeline_report.json"
    (OUT / name).write_text(json.dumps(rep, indent=2, default=float))
    log.info("pipeline.done %s", json.dumps({k: rep[k] for k in ("agent3", "timing") if k in rep}
                                            | {"deterministic": deterministic}, default=float)[:1500])


if __name__ == "__main__":
    main()
