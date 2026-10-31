"""Level 2: Agent 4 (temporal correlation), Agent 5 (adaptive defense), replay loop and the
offline portal validator. Unit tests are model-free; one integration test uses the real agents."""
from pathlib import Path

import pandas as pd
import pytest

from level2 import replay_loop
from src.agents.adaptive_defense import AdaptiveDefense, DefenseConfig
from src.agents.temporal_correlation import CorrelationConfig, TemporalCorrelator, flow_timestamp
from validate_submission import validate

ROOT = Path(__file__).resolve().parents[1]
STRONG = {"decision": "BLOCK", "rule": "attack_agreement", "final_class": "DOS", "evidence_status": "AGREEMENT"}
WEAK = {"decision": "BLOCK", "rule": "low_confidence", "final_class": "BENIGN", "evidence_status": "UNCERTAIN"}
NORMAL = {"decision": "ALLOW", "rule": "benign_agreement", "final_class": "BENIGN", "evidence_status": "AGREEMENT"}
CONFLICT = {"decision": "ALLOW", "rule": "benign_over_anomaly", "final_class": "BENIGN", "evidence_status": "DISAGREEMENT"}


def F(seq, src="10.0.0.9", dst="10.0.0.50", port=80, day="2017-07-04", t=None, proto=6):
    return {"record_id": f"F{seq:06d}", "seq": seq, "source_ip": src, "destination_ip": dst,
            "destination_port": port, "protocol": proto, "day": day,
            "time_offset": t if t is not None else f"{seq // 60:02d}:{seq % 60:02d}.0"}


def run(flows, a3s, **kw):
    rows, defense, corr, trace = replay_loop(flows, a3s, **kw)
    return [(a, p) for _, a, p in rows], defense.ledger(), corr.report()


# --------------------------------------------------------------- Agent 4
def test_timestamp_parsing_and_day_boundary():
    a, b = flow_timestamp("2017-07-04", "58:49.9"), flow_timestamp("2017-07-05", "00:10.2")
    assert (b - a).total_seconds() == pytest.approx(86400 - 3519.7)          # next day is later
    assert flow_timestamp("2017-07-04", "1:02:03") == flow_timestamp("2017-07-04", "62:03")
    assert flow_timestamp("2017-07-04", "garbage") is None and flow_timestamp("bad", "00:01.0") is None


def test_multi_port_weak_flows_form_source_scope_group():                          # 1
    flows = [F(1, port=22), F(2, port=23), F(3, port=80), F(4, port=443)]
    acts, led, groups = run(flows, [WEAK] * 4)
    assert acts[:2] == [("ALLOW", "NONE"), ("ALLOW", "NONE")]                       # 1 weak flow is not enough
    assert led[0]["scope_type"] == "SOURCE_IP" and led[0]["activated_after_record_id"] == "F000002"
    assert acts[2:] == [("BLOCK", "P0001")] * 2


def test_repeated_strong_pair_blocked_after_first_flow():                         # 2, 11, 15, 16
    acts, led, groups = run([F(1), F(2), F(3)], [STRONG] * 3)
    assert acts == [("ALLOW", "NONE"), ("BLOCK", "P0001"), ("BLOCK", "P0001")]      # trigger itself allowed
    assert led[0]["scope_type"] == "SRC_DST_PAIR" and led[0]["scope_value"] == "10.0.0.9|10.0.0.50"
    assert groups[0]["evidence_strength"] == "MEDIUM" and groups[0]["member_records"] == ["F000001"]


def test_same_destination_multiple_sources_get_separate_groups():                 # 3
    flows = [F(1, src="10.0.0.1"), F(2, src="10.0.0.2"), F(3, src="10.0.0.3", port=53), F(4, src="10.0.0.1")]
    acts, led, groups = run(flows, [STRONG, STRONG, NORMAL, STRONG])
    assert [g["source_ip"] for g in groups] == ["10.0.0.1", "10.0.0.2"]
    assert acts[2] == ("ALLOW", "NONE") and acts[3] == ("BLOCK", "P0001")


def test_normal_and_high_volume_benign_traffic_never_blocked():                   # 4, 5, 12
    acts, led, groups = run([F(i) for i in range(1, 61)], [NORMAL] * 60)
    assert set(acts) == {("ALLOW", "NONE")} and led == [] and groups == []


def test_different_days_do_not_merge():                                           # 6
    flows = [F(1, day="2017-07-04", t="10:00.0"), F(2, day="2017-07-05", t="10:00.0")]
    acts, led, groups = run(flows, [WEAK, WEAK])
    assert len(groups) == 2 and led == []


@pytest.mark.parametrize("t2,merged", [("05:00.0", True), ("05:00.1", False)])
def test_time_window_boundary(t2, merged):                                        # 7
    acts, led, groups = run([F(1, t="00:00.0"), F(2, t=t2), F(3, src="10.9.9.9", t="09:00.0")], [WEAK, WEAK, NORMAL])
    assert (len(groups) == 1) == merged and (len(led) == 1) == merged


def test_out_of_order_and_duplicate_records_rejected():                           # 8, 9
    c = TemporalCorrelator()
    c.observe(F(5), STRONG)
    with pytest.raises(ValueError):
        c.observe(F(4), STRONG)
    c2 = TemporalCorrelator()
    c2.observe({**F(1), "record_id": "X"}, NORMAL)
    with pytest.raises(ValueError):
        c2.observe({**F(2), "record_id": "X"}, NORMAL)
    with pytest.raises(ValueError):
        AdaptiveDefense(["F1", "F1"])


def test_missing_ip_cannot_correlate_or_trigger():                               # 10
    acts, led, groups = run([F(1, src=None), F(2, src="")], [STRONG, STRONG])
    assert led == [] and set(acts) == {("ALLOW", "NONE")}


def test_agent3_conflict_is_not_suspicious_and_uncertain_needs_corroboration():   # 13, 14
    assert run([F(1), F(2)], [CONFLICT, CONFLICT])[1] == []
    assert run([F(1)], [WEAK])[1] == []
    assert len(run([F(1), F(2), F(3, src="10.9.9.9")], [WEAK, WEAK, NORMAL])[1]) == 1


# --------------------------------------------------------------- Agent 5 + replay rules
def test_expiry_inclusive_then_reanalysed():
    flows = [F(i) for i in range(1, 7)]
    acts, led, _ = run(flows, [STRONG] * 6, def_cfg=DefenseConfig(ttl_records=3))
    assert led[0]["expires_after_record_id"] == "F000004"
    assert acts == [("ALLOW", "NONE"), ("BLOCK", "P0001"), ("BLOCK", "P0001"), ("BLOCK", "P0001"),
                    ("ALLOW", "NONE"), ("BLOCK", "P0002")]                         # F5 re-analysed -> P0002


def test_smallest_matching_policy_reported():
    d = AdaptiveDefense([f"F{i:06d}" for i in range(1, 6)])
    d.policies = [{"policy_id": "P0002", "scope_type": "SOURCE_IP", "scope_value": "10.0.0.9", "_act": 1, "_exp": 5},
                  {"policy_id": "P0001", "scope_type": "SRC_DST_PAIR", "scope_value": "10.0.0.9|10.0.0.50", "_act": 2, "_exp": 5}]
    assert d.check(F(2)) == ("BLOCK", "P0002") and d.check(F(3)) == ("BLOCK", "P0001")


def test_no_policy_after_last_record():
    acts, led, _ = run([F(1, port=53), F(2)], [NORMAL, STRONG])
    assert led == []


def test_prefix_replay_never_depends_on_future_records():
    flows = [F(1, port=22), F(2), F(3, src="10.0.0.7"), F(4), F(5, port=23), F(6), F(7, src="10.0.0.7"), F(8)]
    a3s = [WEAK, NORMAL, STRONG, WEAK, WEAK, STRONG, STRONG, NORMAL]
    full, _, _ = run(flows, a3s, def_cfg=DefenseConfig(ttl_records=100))
    for k in range(1, len(flows) + 1):
        assert run(flows[:k], a3s[:k], def_cfg=DefenseConfig(ttl_records=100))[0] == full[:k]


# --------------------------------------------------------------- validator
def _sub(flows, a3s):
    rows, defense, _, _ = replay_loop(flows, a3s)
    recs = pd.DataFrame(flows)
    cls = pd.DataFrame({"record_id": recs.record_id, "predicted_class": "DOS", "anomaly_score": "0.500000",
                        "evidence_status": "AGREE"})
    rep = pd.DataFrame(rows, columns=["record_id", "action", "active_policy_id"])
    pol = pd.DataFrame(defense.ledger(), columns=["policy_id", "activated_after_record_id", "scope_type",
                                                  "scope_value", "expires_after_record_id"])
    return recs, cls, rep, pol


def test_validator_accepts_engine_output_and_rejects_tampering():
    recs, cls, rep, pol = _sub([F(i) for i in range(1, 6)], [STRONG] * 5)
    assert validate(recs, cls, rep, pol) == []
    bad = rep.copy(); bad.loc[1, ["action", "active_policy_id"]] = ["ALLOW", "NONE"]
    assert validate(recs, cls, bad, pol)
    assert validate(recs, cls.iloc[:-1], rep, pol)                                  # missing row
    assert validate(recs, cls.assign(evidence_status="AGREEMENT"), rep, pol)        # bad status
    assert validate(recs, cls.assign(anomaly_score="1.5"), rep, pol)
    assert validate(recs, cls, rep, pol.assign(scope_value="10.0.0.0/24"))          # CIDR not allowed
    assert validate(recs, cls, rep, pol.assign(policy_action="BLOCK"))              # header must match


# --------------------------------------------------------------- integration (real agents, hidden set)
HIDDEN = ROOT / "data" / "hidden" / "Cyber" / "hidden" / "records.csv"


@pytest.mark.skipif(not HIDDEN.exists() or not (ROOT / "models" / "anomaly_detector.joblib").exists(),
                    reason="hidden records or Agent 2 model not present")
def test_hidden_replay_is_valid_and_deterministic():
    from level2 import replay
    recs = pd.read_csv(HIDDEN, dtype={"record_id": str})
    out1, out2 = replay(recs), replay(recs.sample(frac=1, random_state=0))          # shuffled input is re-sorted
    for x, y in zip(out1[:3], out2[:3]):
        pd.testing.assert_frame_equal(x.reset_index(drop=True), y.reset_index(drop=True))
    cls, rep, pol, _ = out1
    assert len(cls) == len(rep) == len(recs) == 200
    assert validate(recs, cls.astype(str), rep, pol) == []
