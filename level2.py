"""CyberSentinel Level 2: five-agent chronological replay -> the three submission CSVs.

  python level2.py                                         # official hidden records
  python level2.py --input path/records.csv --out outputs/level2

A1 (class) + A2 (anomaly) -> A3 (evidence) for every flow; then, in seq order, A5 applies
already-active policies first; only ALLOWed flows go to A4 (temporal correlation) and may
lead A5 to create a policy for LATER records. A1-A3 score each flow independently from its
own features, so scoring them in one batch uses no information from other records.
Writes classification.csv, replay.csv, policies.csv + agent logs, then validates them.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from run_pipeline import run
from src.agents.adaptive_defense import AdaptiveDefense, DefenseConfig
from src.agents.temporal_correlation import CorrelationConfig, TemporalCorrelator
from validate_submission import validate

ROOT = Path(__file__).resolve().parent
STATUS = {"AGREEMENT": "AGREE", "DISAGREEMENT": "CONFLICT", "UNCERTAIN": "UNCERTAIN", "INVALID_EVIDENCE": "UNCERTAIN"}
log = logging.getLogger("cybersentinel.level2")


def replay_loop(flows: list[dict], a3: list[dict], corr_cfg=CorrelationConfig(), def_cfg=DefenseConfig()):
    """The chronological A5 -> A4 -> A5 loop over flows already sorted by seq (model-free, testable)."""
    corr, defense = TemporalCorrelator(corr_cfg), AdaptiveDefense([f["record_id"] for f in flows], def_cfg)
    rows, trace = [], []
    for flow, r3 in zip(flows, a3):
        action, pid = defense.check(flow)
        step = {"seq": flow["seq"], "record_id": flow["record_id"], "source_ip": flow.get("source_ip"),
                "destination_ip": flow.get("destination_ip"), "destination_port": flow.get("destination_port"),
                "agent3": {k: r3.get(k) for k in ("final_class", "evidence_status", "decision", "rule")},
                "action": action, "active_policy_id": pid}
        if action == "ALLOW":                      # only allowed flows are analysed (brief rule)
            c = corr.observe(flow, r3)
            p = defense.consider(flow, c)
            step.update(agent4_group=c["group_id"], agent4_strength=c["evidence_strength"],
                        new_policy=p["policy_id"] if p else None)
        rows.append((flow["record_id"], action, pid))
        trace.append(step)
    return rows, defense, corr, trace


def replay(records: pd.DataFrame, corr_cfg=CorrelationConfig(), def_cfg=DefenseConfig()):
    df = records.sort_values("seq", kind="stable").reset_index(drop=True)
    if df.record_id.duplicated().any() or df.seq.duplicated().any():
        raise ValueError("duplicate record_id or seq in replay input")
    a1, a2, a3, _ = run(df)
    bad = [r for r, x, y in zip(df.record_id, a1, a2) if not isinstance(x, dict) or not isinstance(y, dict)]
    if bad:
        raise RuntimeError(f"agent failure on {len(bad)} record(s), first {bad[0]}; refusing to fabricate scores")
    cls = pd.DataFrame({"record_id": df.record_id,
                        "predicted_class": [r["final_class"] if r["final_class"] != "UNKNOWN" else "BENIGN" for r in a3],
                        "anomaly_score": [f"{r['anomaly_score']:.6f}" for r in a2],
                        "evidence_status": [STATUS[r["evidence_status"]] for r in a3]})
    rows, defense, corr, trace = replay_loop(df.to_dict("records"), a3, corr_cfg, def_cfg)
    rep = pd.DataFrame(rows, columns=["record_id", "action", "active_policy_id"])
    pol = pd.DataFrame(defense.ledger(), columns=["policy_id", "activated_after_record_id", "scope_type",
                                                   "scope_value", "expires_after_record_id"])
    return cls, rep, pol, {"groups": corr.report(), "policies": defense.ledger(), "trace": trace,
                           "agent3": a3, "agent1": a1, "agent2": a2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data" / "hidden" / "Cyber" / "hidden" / "records.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "level2")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("src.agents.evidence_fusion").setLevel(logging.ERROR)
    records = pd.read_csv(a.input, dtype={"record_id": str})
    cls, rep, pol, logs = replay(records)
    a.out.mkdir(parents=True, exist_ok=True)
    cls.to_csv(a.out / "classification.csv", index=False)
    rep.to_csv(a.out / "replay.csv", index=False)
    pol.to_csv(a.out / "policies.csv", index=False)
    (a.out / "agent4_groups.json").write_text(json.dumps(logs["groups"], indent=1, default=str))
    (a.out / "agent5_policies.json").write_text(json.dumps(logs["policies"], indent=1, default=str))
    (a.out / "replay_trace.jsonl").write_text("\n".join(json.dumps(s, default=str) for s in logs["trace"]) + "\n")
    read = lambda n: pd.read_csv(a.out / n, dtype=str, keep_default_na=False)
    errors = validate(records, read("classification.csv"), read("replay.csv"), read("policies.csv"))
    log.info("level2.done rows=%d policies=%d blocks=%d classes=%s valid=%s", len(cls), len(pol),
             int((rep.action == "BLOCK").sum()), cls.predicted_class.value_counts().to_dict(), not errors)
    for e in errors[:20]:
        log.error("validator: %s", e)


if __name__ == "__main__":
    main()
