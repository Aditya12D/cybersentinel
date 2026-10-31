"""Offline copy of the Level 2 portal validator rules (stage page, 31 Oct).

  python validate_submission.py [records.csv] [submission_dir]

Recomputes every replay row from policies.csv exactly as the portal describes:
  matches = policies where activatedIndex < i <= expiresIndex and scope matches the flow
  action = BLOCK if matches else ALLOW; active_policy_id = min(matches) or NONE
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

from src.agents.adaptive_defense import scope_matches

HEADERS = {"classification.csv": ["record_id", "predicted_class", "anomaly_score", "evidence_status"],
           "replay.csv": ["record_id", "action", "active_policy_id"],
           "policies.csv": ["policy_id", "activated_after_record_id", "scope_type", "scope_value", "expires_after_record_id"]}
CLASSES = {"BENIGN", "BRUTE_FORCE", "DOS", "DDOS", "PORT_SCAN"}
STATUSES, ACTIONS = {"AGREE", "CONFLICT", "UNCERTAIN"}, {"ALLOW", "BLOCK"}
SCOPES = {"SOURCE_IP", "DESTINATION_IP", "SRC_DST_PAIR"}


def validate(records: pd.DataFrame, cls: pd.DataFrame, rep: pd.DataFrame, pol: pd.DataFrame) -> list[str]:
    err = []
    for name, df in (("classification.csv", cls), ("replay.csv", rep), ("policies.csv", pol)):
        if list(df.columns) != HEADERS[name]:
            err.append(f"{name}: header {list(df.columns)} != {HEADERS[name]}")
    if err:
        return err
    recs = records.sort_values("seq").reset_index(drop=True)
    ids = recs.record_id.astype(str).tolist()
    pos = {r: i for i, r in enumerate(ids, start=1)}
    for name, df in (("classification.csv", cls), ("replay.csv", rep)):
        got = df.record_id.astype(str)
        if len(df) != len(ids) or got.duplicated().any() or set(got) != set(ids):
            err.append(f"{name}: must have exactly one row per hidden record_id ({len(df)} rows)")
    err += [f"classification.csv: bad predicted_class {v!r}" for v in set(cls.predicted_class) - CLASSES]
    err += [f"classification.csv: bad evidence_status {v!r}" for v in set(cls.evidence_status) - STATUSES]
    for rid, s in zip(cls.record_id, cls.anomaly_score):
        try:
            ok = math.isfinite(float(s)) and 0.0 <= float(s) <= 1.0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            err.append(f"classification.csv: anomaly_score {s!r} for {rid} not a number in [0, 1]")
    err += [f"replay.csv: bad action {v!r}" for v in set(rep.action) - ACTIONS]
    if pol.policy_id.duplicated().any():
        err.append("policies.csv: duplicate policy_id")
    srcs, dsts = set(recs.source_ip), set(recs.destination_ip)
    pairs = set(recs.source_ip + "|" + recs.destination_ip)
    observed = {"SOURCE_IP": srcs, "DESTINATION_IP": dsts, "SRC_DST_PAIR": pairs}
    for p in pol.itertuples():
        if p.activated_after_record_id not in pos or p.expires_after_record_id not in pos:
            err.append(f"policies.csv: {p.policy_id} references unknown record_id")
        elif pos[p.activated_after_record_id] > pos[p.expires_after_record_id]:
            err.append(f"policies.csv: {p.policy_id} expires before it activates")
        if p.scope_type not in SCOPES:
            err.append(f"policies.csv: {p.policy_id} bad scope_type {p.scope_type!r}")
        elif p.scope_value not in observed[p.scope_type]:
            err.append(f"policies.csv: {p.policy_id} scope_value {p.scope_value!r} is not an observed endpoint")
    if err:
        return err
    want = {}
    for i, f in enumerate(recs.to_dict("records"), start=1):
        m = sorted(p.policy_id for p in pol.itertuples()
                   if pos[p.activated_after_record_id] < i <= pos[p.expires_after_record_id]
                   and scope_matches(p.scope_type, p.scope_value, f))
        want[str(f["record_id"])] = ("BLOCK", m[0]) if m else ("ALLOW", "NONE")
    for r in rep.itertuples():
        if (r.action, str(r.active_policy_id)) != want[str(r.record_id)]:
            err.append(f"replay.csv: {r.record_id} is {r.action},{r.active_policy_id}; ledger implies {want[str(r.record_id)]}")
    return err


def main() -> None:
    records = Path(sys.argv[1] if len(sys.argv) > 1 else "data/hidden/Cyber/hidden/records.csv")
    sub = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/level2")
    read = lambda n: pd.read_csv(sub / n, dtype=str, keep_default_na=False)
    errors = validate(pd.read_csv(records, dtype={"record_id": str}), read("classification.csv"),
                      read("replay.csv"), read("policies.csv"))
    print("VALID" if not errors else f"INVALID ({len(errors)} problems)")
    for e in errors[:50]:
        print(" -", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
