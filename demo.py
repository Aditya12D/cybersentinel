"""CyberSentinel five-agent demo (terminal, ASCII only).

  python demo.py          # interactive menu
  python demo.py 7        # one section, 1-8
  python demo.py all      # every section, no prompts

Every number is computed live by the real agents on the official hidden replay (no labels, no
future records). Nothing here changes agent logic or the frozen submission; scenarios absent from
the hidden replay are taken from the labelled validation set or marked as TEST FIXTURE.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import pandas as pd

from level2 import replay
from run_pipeline import run
from src.agents.adaptive_defense import AdaptiveDefense
from src.agents.anomaly_detection.api import load_detector
from src.agents.evidence_fusion import ATTACKS, DEFAULT_CONFIG, fuse
from src.agents.temporal_correlation import CorrelationConfig, TemporalCorrelator, flow_timestamp

ROOT = Path(__file__).resolve().parent
HIDDEN = ROOT / "data" / "hidden" / "Cyber" / "hidden" / "records.csv"
EVAL = ROOT / "data" / "raw" / "eval" / "flows.parquet"
FROZEN = ROOT / "submission" / "level2_v1"
W = 100
STATUS = {"AGREEMENT": "AGREE", "DISAGREEMENT": "CONFLICT", "UNCERTAIN": "UNCERTAIN", "INVALID_EVIDENCE": "UNCERTAIN"}


# ------------------------------------------------------------------ printing helpers
def head(title):
    print("\n" + "=" * W + f"\n  {title}\n" + "=" * W)


def sub(title):
    print(f"\n-- {title} " + "-" * max(3, W - len(title) - 4))


def say(*paras):
    for p in paras:
        print(textwrap.fill(p, W, initial_indent="  ", subsequent_indent="  "))


def table(rows, cols):
    rows = [[("" if v is None else str(v)) for v in r] for r in rows]
    w = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    line = lambda vals: "  " + "  ".join(v.ljust(w[i]) for i, v in enumerate(vals))
    print(line(cols)); print("  " + "  ".join("-" * x for x in w))
    for r in rows:
        print(line(r))


# ------------------------------------------------------------------ live state (computed once)
@lru_cache(maxsize=1)
def state():
    if not HIDDEN.exists():
        sys.exit(f"Hidden records not found at {HIDDEN}. See docs/DEMO_SETUP.md.")
    records = pd.read_csv(HIDDEN, dtype={"record_id": str})
    cls, rep, pol, logs = replay(records)                       # the real five-agent replay
    flows = records.sort_values("seq", kind="stable").to_dict("records")
    by_id = {f["record_id"]: i for i, f in enumerate(flows)}
    return {"records": records, "flows": flows, "by_id": by_id, "cls": cls, "rep": rep, "pol": pol, **logs}


def rec(rid):
    s = state(); i = s["by_id"][rid]
    return s["flows"][i], s["agent1"][i], s["agent2"][i], s["agent3"][i], s["trace"][i]


def first(pred):
    s = state()
    return next((f["record_id"] for i, f in enumerate(s["flows"]) if pred(i)), None)


# ------------------------------------------------------------------ 1. Agent 1
def agent1():
    head("AGENT 1 - THREAT CLASSIFICATION  (question: what type of traffic is this flow?)")
    meta = json.loads((ROOT / "models" / "model_metadata.json").read_text())
    say("Input: one flow's numeric CICFlowMeter statistics (packet counts and sizes, timings, TCP flags, "
        "rates). Endpoints, ports-as-metadata, time and labels are NOT inputs.",
        f"Model: LightGBM gradient-boosted trees, {meta['n_estimators']} trees, trained on "
        f"{meta['train_rows']:,} labelled training flows (5 classes, no class weights, seed {meta['seed']}).",
        f"Features: 78 numeric columns -> {len(meta['features'])} used. The {len(meta['dropped_features'])} "
        "dropped are zero-information columns only: 3 constant in training (URG flags) and 4 exact copies of "
        "another column. No PCA / RFE / importance pruning.",
        "Output: a probability for each of BENIGN, BRUTE_FORCE, DOS, DDOS, PORT_SCAN (they sum to 1); the "
        "predicted class is the most probable one and 'confidence' is that probability.")
    sub("Real outputs on the hidden replay (first record of each predicted class, in seq order)")
    s = state(); rows = []
    for c in ("BENIGN", *ATTACKS):
        rid = first(lambda i: s["agent1"][i]["predicted_class"] == c)
        if rid:
            _, a1, *_ = rec(rid)
            rows.append([rid, a1["predicted_class"], f"{a1['confidence']:.6f}",
                         *(f"{a1['probabilities'][k]:.6f}" for k in ("BENIGN", *ATTACKS))])
    table(rows, ["record", "predicted", "confidence", "P(BENIGN)", "P(BRUTE)", "P(DOS)", "P(DDOS)", "P(PSCAN)"])
    say("", "These are predictions, not ground truth: the hidden set has no labels.")


# ------------------------------------------------------------------ 2. Agent 2
def agent2():
    head("AGENT 2 - ANOMALY DETECTION  (question: how unusual is this flow compared with normal traffic?)")
    d = load_detector()
    say("Why a second agent: Agent 1 only knows the 5 classes it was trained on. Agent 2 is trained on BENIGN "
        "traffic only and measures distance from normal, so it can notice unusual traffic of any kind - but it "
        "never names an attack type.",
        f"How: {len(d.model_features)} de-correlated features (log + robust scaling). Two complementary members: "
        "k-nearest-neighbour distance to 100,000 normal prototypes (catches DoS/DDoS/brute force) and an "
        "Isolation Forest (catches port scans). Each is turned into a percentile of held-out normal flows and "
        "the two are fused.",
        f"anomaly_score = fraction of held-out NORMAL flows that are less unusual. 0.98 means 'more unusual than "
        f"98% of normal traffic'. It is NOT a probability of attack. Threshold {d.threshold} = a 3% "
        "false-alarm budget on normal traffic by construction.",
        "Data quality: missing / infinite / non-numeric values are repaired and listed; if more than 10% of "
        "features were repaired the result is marked reliable=false (Agent 3 then treats it as UNCERTAIN).")
    s = state()
    picks = [("A1 attack + flagged", first(lambda i: s["agent3"][i]["rule"] == "attack_agreement")),
             ("A1 benign + flagged", first(lambda i: s["agent3"][i]["rule"] == "benign_over_anomaly")),
             ("A1 benign + normal", first(lambda i: s["agent3"][i]["rule"] == "benign_agreement"))]
    sub("Real outputs on the hidden replay")
    rows = []
    for label, rid in picks:
        if rid:
            _, _, a2, _, _ = rec(rid)
            rows.append([rid, label, f"{a2['anomaly_score']:.6f}", a2["is_anomaly"], a2["data_quality"]["reliable"],
                         a2["components"]["knn_benign_percentile"], a2["components"]["isolation_forest_benign_percentile"]])
    table(rows, ["record", "case", "anomaly_score", "is_anomaly", "reliable", "kNN pct", "IForest pct"])
    rid = picks[0][1]
    if rid:
        _, _, a2, _, _ = rec(rid)
        sub(f"Feature evidence for {rid} (measured values vs the normal reference)")
        table([[e["feature"], e["observed_value"], e["severity"], textwrap.shorten(e["reference"], 60)]
               for e in a2["evidence"][:4]], ["feature", "observed", "severity", "reference"])
        say("", "Agent 2's own explanation: " + a2["reason"])


# ------------------------------------------------------------------ 3. Agent 3
@lru_cache(maxsize=1)
def validation_examples():
    """Rules that never fire on the hidden replay, taken from the labelled validation set."""
    if not EVAL.exists():
        return {}
    ev = pd.read_parquet(EVAL)
    out = {}
    for cls_name, rule in (("DOS", "attack_over_normal"), ("PORT_SCAN", "low_confidence")):
        part = ev[ev.label == cls_name]
        a1, a2, fz, _ = run(part)
        i = next((k for k, r in enumerate(fz) if r["rule"] == rule), None)
        if i is not None:
            out[rule] = (part.iloc[i].record_id, a1[i], a2[i], fz[i])
    return out


def agent3():
    head("AGENT 3 - EVIDENCE FUSION  (question: do the two independent signals support each other?)")
    say("Agent 1 (a classifier) and Agent 2 (an unusualness detector) answer different questions, so they are "
        "not interchangeable votes. Agent 3 first validates both outputs strictly, then applies these rules in "
        "order (deterministic, no LLM):")
    table([["1", "Agent 1 output invalid / missing / crashed", "UNKNOWN", "BLOCK", "UNCERTAIN"],
           ["2", "Agent 2 output invalid / missing / crashed", "Agent 1 class", "BLOCK", "UNCERTAIN"],
           ["3", "Agent 2 says its result is unreliable", "Agent 1 class", "BLOCK", "UNCERTAIN"],
           ["4", f"Agent 1 confidence < {DEFAULT_CONFIG.min_confidence}", "Agent 1 class", "BLOCK", "UNCERTAIN"],
           ["5", "A1 predicts attack + A2 flags anomalous", "Agent 1 class", "BLOCK", "AGREE"],
           ["6", "A1 predicts attack + A2 normal", "Agent 1 class", "BLOCK", "CONFLICT"],
           ["7", "A1 confident BENIGN + A2 flags anomalous", "BENIGN", "ALLOW + review", "CONFLICT"],
           ["8", "A1 confident BENIGN + A2 normal", "BENIGN", "ALLOW", "AGREE"]],
          ["#", "condition", "final class", "initial decision", "csv status"])
    say("", "Agent 2 can never supply or change an attack class; an Agent 2 'normal' never downgrades an Agent 1 "
        "attack; invalid evidence never produces ALLOW.")
    s = state(); rows = []
    sub("Real hidden-replay examples (every rule that occurs in the 200 records)")
    for rule in ("attack_agreement", "attack_over_normal", "benign_over_anomaly", "benign_agreement",
                 "low_confidence", "unreliable_anomaly"):
        rid = first(lambda i: s["agent3"][i]["rule"] == rule)
        if rid:
            _, a1, a2, a3, _ = rec(rid)
            rows.append([rid, rule, f"{a1['predicted_class']} {a1['confidence']:.4f}",
                         f"{a2['anomaly_score']:.3f} {'flag' if a2['is_anomaly'] else 'normal'}",
                         STATUS[a3["evidence_status"]], a3["decision"]])
    table(rows, ["record", "rule", "Agent 1", "Agent 2", "status", "decision"])
    sub("Rules that do not occur in the hidden replay")
    ex, rows = validation_examples(), []
    for rule, (rid, a1, a2, a3) in ex.items():
        rows.append([f"{rid} (validation set)", rule, f"{a1['predicted_class']} {a1['confidence']:.4f}",
                     f"{a2['anomaly_score']:.3f} {'flag' if a2['is_anomaly'] else 'normal'}",
                     STATUS[a3["evidence_status"]], a3["decision"]])
    f0 = s["flows"][s["by_id"][first(lambda i: s["agent3"][i]["rule"] == "attack_agreement")]]
    damaged = pd.DataFrame([f0])
    feats = [c for c in damaged.columns if c not in ("record_id", "seq", "day", "time_offset", "source_ip",
                                                       "destination_ip", "source_port", "destination_port", "protocol")]
    damaged.loc[0, feats[::2]] = float("nan")                  # TEST FIXTURE: half the features missing
    a1d, a2d, fzd, _ = run(damaged)
    rows.append([f"{f0['record_id']} with 50% features blanked (TEST FIXTURE)", fzd[0]["rule"],
                 f"{a1d[0]['predicted_class']} {a1d[0]['confidence']:.4f}",
                 f"reliable={a2d[0]['data_quality']['reliable']}", STATUS[fzd[0]["evidence_status"]], fzd[0]["decision"]])
    crash = fuse(s["agent1"][0], RuntimeError("anomaly service down"), record_id=s["flows"][0]["record_id"])
    rows.append([f"{s['flows'][0]['record_id']} + Agent 2 crash (TEST FIXTURE)", crash["rule"],
                 s["agent1"][0]["predicted_class"], "raised error", STATUS[crash["evidence_status"]], crash["decision"]])
    table(rows, ["record", "rule", "Agent 1", "Agent 2", "status", "decision"])
    rid = first(lambda i: s["agent3"][i]["rule"] == "benign_over_anomaly")
    if rid:
        say("", f"Agent 3 rationale for {rid}: " + rec(rid)[3]["rationale"])


# ------------------------------------------------------------------ 4. Agent 4
def agent4():
    head("AGENT 4 - TEMPORAL CORRELATION  (question: which suspicious flows belong together over time?)")
    cfg = CorrelationConfig()
    f = state()["flows"][0]
    say("Single flows can look harmless; a burst from one source across ports or time is a stronger signal. "
        "Agent 4 groups suspicious flows (Agent 3 recommends BLOCK) by source while consecutive flows are within "
        f"{cfg.window_seconds:.0f} s. It sees only the current and earlier records, and never blocks anything.",
        f"Time: day + time_offset -> real timestamp, e.g. {f['record_id']}: {f['day']} + {f['time_offset']} = "
        f"{flow_timestamp(f['day'], f['time_offset'])}. Different days never merge; strings are never compared.",
        "Evidence strength: HIGH = 2+ flows incl. one where Agent 1 predicted an attack and Agent 2 flagged it; "
        "MEDIUM = one such flow, or 2+ weaker flows; LOW = one weak flow (never enough to act on).",
        f"It also proposes the policy window (activation = this record, expiry = +{cfg.policy_window_records} "
        "records) and counts destination fan-in (distinct suspicious sources hitting one destination).")
    s = state()
    sub("Groups in the real replay (Agent 4 only sees flows that existing policies did not block)")
    table([[g["group_id"], g["source_ip"], g["destination_ip"] or f"{g['unique_destinations']} hosts",
            g["unique_destination_ports"], g["suspicious_flow_count"], ",".join(g["member_records"][:4]),
            g["time_window_seconds"], g["pattern"], g["evidence_strength"],
            f"after seq {g['first_seq']} -> seq {g['first_seq'] + cfg.policy_window_records}"
            + (" (A5 caps at last record)" if g["first_seq"] + cfg.policy_window_records > len(s["flows"]) else "")]
           for g in s["groups"]],
          ["group", "source", "destination", "ports", "flows", "records", "span s", "pattern", "strength", "proposed window"])
    sub("Illustration: Agent 4 run on ALL 200 flows with no blocking (shows how groups grow)")
    c = TemporalCorrelator(cfg); fan = 0
    for flow, a3 in zip(s["flows"], s["agent3"]):
        r = c.observe(flow, a3); fan = max(fan, r.get("destination_fan_in") or 0)
    table([[g["group_id"], g["source_ip"], g["unique_destinations"], g["unique_destination_ports"],
            g["suspicious_flow_count"], f"{g['first_seq']}-{g['last_seq']}", g["time_window_seconds"],
            g["pattern"], g["evidence_strength"]] for g in c.report()],
          ["group", "source", "dests", "ports", "flows", "seq range", "span s", "pattern", "strength"])
    say("", f"Max destination fan-in: {fan} distinct suspicious source(s) on one destination. In the real replay "
        "most of these flows are blocked first, so Agent 4 never needs to grow these groups.")


# ------------------------------------------------------------------ 5. Agent 5
def agent5():
    head("AGENT 5 - ADAPTIVE DEFENSE  (question: should later traffic be blocked, and how narrowly?)")
    say("Classification alone cannot protect later traffic. Agent 5 is the only component that creates or "
        "enforces policies. HIGH or MEDIUM evidence from Agent 4 makes it enact Agent 4's proposed window as a "
        "temporary BLOCK policy.",
        "Scope: SRC_DST_PAIR (source|destination) while evidence involves one host and one port - narrowest, keeps "
        "the source's other traffic; SOURCE_IP once the source touched several ports or hosts. DESTINATION_IP is "
        "never used by Agent 5 because it would block every legitimate client of that server.",
        "Timing: active from the record AFTER the trigger (exclusive) through the expiry record (inclusive). A "
        "policy cannot block its own trigger. For every new record, active policies are checked FIRST; a blocked "
        "flow is not analysed further. A BLOCK reports the lexicographically smallest matching policy id.")
    s = state(); rep = s["rep"]
    sub("Policy ledger from the real replay")
    table([[p["policy_id"], p["activated_after_record_id"], p["scope_type"], p["scope_value"],
            p["expires_after_record_id"], int((rep.active_policy_id == p["policy_id"]).sum()),
            p["group_id"], p["evidence_strength"]] for p in s["policies"]],
          ["policy", "activated after", "scope", "value", "expires after (incl.)", "records blocked", "from group", "evidence"])
    sub("TEST FIXTURE: two overlapping policies -> smallest id is reported")
    d = AdaptiveDefense([f"F{i:06d}" for i in range(1, 6)])
    d.policies = [{"policy_id": "P0002", "scope_type": "SOURCE_IP", "scope_value": "10.0.0.9", "_act": 1, "_exp": 5},
                  {"policy_id": "P0001", "scope_type": "SRC_DST_PAIR", "scope_value": "10.0.0.9|10.0.0.50", "_act": 2, "_exp": 5}]
    flow = lambda k: {"record_id": f"F{k:06d}", "source_ip": "10.0.0.9", "destination_ip": "10.0.0.50"}
    table([[f"F{k:06d}", *d.check(flow(k))] for k in (2, 3)], ["record", "action", "active_policy_id"])
    say("", "F000002: only P0002 is active yet. F000003: both match, so P0001 (smaller) is reported.")


# ------------------------------------------------------------------ 6. whole pipeline
DIAGRAM = r"""
   hidden record (seq order)
          |
          +--------------------+---------------------+
          v                                          v
   AGENT 1  Threat Classification            AGENT 2  Anomaly Detection
   class + confidence + 5 probabilities      anomaly_score + flag + evidence + data quality
          |                                          |
          +--------------------+---------------------+
                               v
                AGENT 3  Evidence Fusion  -> final class, AGREE/CONFLICT/UNCERTAIN, initial decision
                               |                      (-> classification.csv for every record)
                               v
   +-------------- REPLAY LOOP, one record at a time, in seq order -----------------------+
   |  AGENT 5 check: does an ACTIVE policy match this flow?                               |
   |     yes -> BLOCK (policy id). Flow is NOT analysed further (Agent 4 skipped).        |
   |     no  -> ALLOW, then                                                               |
   |            AGENT 4 observe: add to source group, evidence strength, proposed window  |
   |            AGENT 5 consider: HIGH/MEDIUM -> create policy for LATER records          |
   +---------------------------------------------------------------------------------------+
                               v
          replay.csv (ALLOW/BLOCK + policy id)    policies.csv (the ledger)
"""


def frozen_check():
    s = state(); rows, ok = [], True
    sums = dict(l.split()[::-1] for l in (FROZEN / "SHA256SUMS").read_text().splitlines()) if (FROZEN / "SHA256SUMS").exists() else {}
    for name, df in (("classification.csv", s["cls"]), ("replay.csv", s["rep"]), ("policies.csv", s["pol"])):
        live = hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
        match = live == sums.get(name)
        ok &= match
        rows.append([name, live[:16] + "...", (sums.get(name) or "missing")[:16] + "...", "IDENTICAL" if match else "DIFFERENT"])
    table(rows, ["file", "live run sha256", "frozen sha256", "result"])
    return ok


def pipeline():
    head("COMPLETE FIVE-AGENT PIPELINE  (actual execution order in level2.py)")
    print(DIAGRAM)
    s = state()
    sub("First 20 records of the live replay")
    rows = []
    for i in range(20):
        f, a1, a2, a3, t = s["flows"][i], s["agent1"][i], s["agent2"][i], s["agent3"][i], s["trace"][i]
        rows.append([t["seq"], f["record_id"], f"{f['source_ip']}->{f['destination_ip']}:{f['destination_port']}",
                     a1["predicted_class"], f"{a2['anomaly_score']:.3f}{'*' if a2['is_anomaly'] else ''}",
                     STATUS[a3["evidence_status"]], t["active_policy_id"] if t["action"] == "BLOCK" else "-",
                     "skipped" if t["action"] == "BLOCK" else (t.get("agent4_group") or "-"),
                     t.get("new_policy") or "-", t["action"]])
    table(rows, ["seq", "record", "flow", "A1 class", "A2 score", "A3", "policy hit", "A4 group", "A5 new", "action"])
    say("", "* = Agent 2 flagged the flow as anomalous (score >= 0.97).")
    sub("Live run vs frozen submission (submission/level2_v1)")
    print("  " + ("All three files reproduce the frozen submission byte-for-byte." if frozen_check()
                  else "WARNING: live output differs from the frozen submission (see docs/DEMO_SETUP.md)."))


# ------------------------------------------------------------------ 7. key policy records
def key_records():
    head("POLICY LIFECYCLE ON REAL RECORDS  (derived from the ledger, not hardcoded)")
    s = state(); order = [f["record_id"] for f in s["flows"]]
    frozen = pd.read_csv(FROZEN / "replay.csv", dtype=str).set_index("record_id") if (FROZEN / "replay.csv").exists() else None
    for p in s["policies"]:
        a, e = order.index(p["activated_after_record_id"]), order.index(p["expires_after_record_id"])
        match = lambda r: s["trace"][s["by_id"][r]]["active_policy_id"] == p["policy_id"]
        inwin = [r for r in order[a + 1:e + 1] if match(r)]
        after = next((r for r in order[e + 1:] if s["trace"][s["by_id"][r]]["source_ip"] == s["trace"][a]["source_ip"]), None)
        moments = [("TRIGGER", order[a]), ("FIRST BLOCK", inwin[0] if inwin else None),
                   ("LAST BLOCK IN WINDOW", inwin[-1] if inwin else None), ("AFTER EXPIRY", after)]
        sub(f"{p['policy_id']}  {p['scope_type']} {p['scope_value']}  active after {p['activated_after_record_id']} "
            f"through {p['expires_after_record_id']} (inclusive)")
        for label, rid in moments:
            if rid is None:
                continue
            f, a1, a2, a3, t = rec(rid)
            print(f"\n  [{label}] {rid} (seq {f['seq']})  {f['source_ip']} -> {f['destination_ip']}:{f['destination_port']}")
            if t["action"] == "BLOCK":
                print(f"    Agent 5 check : active policy {t['active_policy_id']} matches -> BLOCK before analysis; Agent 4 skipped")
            else:
                print("    Agent 5 check : no active policy matches -> ALLOW, flow is analysed")
            print(f"    Agent 1       : predicted {a1['predicted_class']} (confidence {a1['confidence']:.6f})")
            print(f"    Agent 2       : anomaly_score {a2['anomaly_score']:.3f} -> "
                  f"{'independently flagged as anomalous' if a2['is_anomaly'] else 'not flagged'}")
            print(f"    Agent 3       : {STATUS[a3['evidence_status']]} ({a3['rule']}), initial decision {a3['decision']}")
            if t["action"] == "ALLOW":
                print(f"    Agent 4       : group {t.get('agent4_group')}, evidence {t.get('agent4_strength')}")
                print(f"    Agent 5       : {'created ' + t['new_policy'] + ' for LATER records' if t.get('new_policy') else 'no new policy'}")
            if frozen is not None:
                fr = frozen.loc[rid]
                print(f"    Final action  : {t['action']} {t['active_policy_id']}   (frozen replay.csv: {fr.action} {fr.active_policy_id})")
    say("", "The trigger flow is always allowed: a policy may only affect records after its trigger. After expiry "
        "the next flow is analysed again, which is why a second policy appears.")


# ------------------------------------------------------------------ 8. dashboard
def dashboard():
    head("RESULTS DASHBOARD  (predictions on unlabelled hidden records - no ground truth available)")
    s = state(); cls, rep, pol = s["cls"], s["rep"], s["pol"]
    blocked = int((rep.action == "BLOCK").sum())
    table([["records processed", len(cls)], ["ALLOW", int((rep.action == "ALLOW").sum())], ["BLOCK", blocked],
           ["policies created", len(pol)], ["records skipped by Agent 4 (blocked first)", blocked],
           ["Agent 4 groups formed", len(s["groups"])]], ["metric", "value"])
    sub("Predicted classes (Agent 3 final class) and evidence status")
    table([[k, v] for k, v in cls.predicted_class.value_counts().items()], ["predicted class", "records"])
    print()
    table([[k, v] for k, v in cls.evidence_status.value_counts().items()], ["evidence status", "records"])
    sub("Policies: scope, blocked records, timeline")
    table([[p.policy_id, p.scope_type, p.scope_value, int((rep.active_policy_id == p.policy_id).sum()),
            f"{p.activated_after_record_id} -> {p.expires_after_record_id}"] for p in pol.itertuples()],
          ["policy", "scope", "value", "blocked", "active window (after -> through)"])
    say("", "Without hidden labels we cannot compute false-positive / false-negative rates. What we can say: "
        f"every BLOCK is justified by a policy, every policy by Agent 4 evidence, and "
        f"{int(((cls.predicted_class == 'BENIGN') & (rep.action == 'BLOCK')).sum())} records predicted BENIGN were "
        f"blocked; {int(((cls.predicted_class != 'BENIGN') & (rep.action == 'ALLOW')).sum())} predicted attacks were "
        "allowed (each a policy trigger).")


SECTIONS = {"1": ("Agent 1 - threat classification", agent1), "2": ("Agent 2 - anomaly detection", agent2),
            "3": ("Agent 3 - evidence fusion", agent3), "4": ("Agent 4 - temporal correlation", agent4),
            "5": ("Agent 5 - adaptive defense", agent5), "6": ("Complete five-agent pipeline", pipeline),
            "7": ("Policy activation / expiry on real records", key_records), "8": ("Results dashboard", dashboard)}


def main(argv):
    if argv and argv[0] == "all":
        for _, fn in SECTIONS.values():
            fn()
        return
    if argv:
        SECTIONS[argv[0]][1]()
        return
    print("Loading agents and replaying the 200 hidden records ..."); state()
    while True:
        print("\n" + "\n".join(f"  {k}. {t}" for k, (t, _) in SECTIONS.items()) + "\n  q. quit")
        choice = input("Choose: ").strip().lower()
        if choice == "q":
            break
        if choice in SECTIONS:
            SECTIONS[choice][1]()


if __name__ == "__main__":
    import logging
    logging.disable(logging.WARNING)
    main(sys.argv[1:])
