"""Agent 3 (evidence fusion): unit tests on synthetic agent outputs + end-to-end
integration on real eval rows (skipped if the Agent 2 model is not trained)."""
import copy
import math
from pathlib import Path

import pytest

from src.agents.evidence_fusion import (ALLOW, ATTACKS, BLOCK, CLASSES, UNKNOWN, FusionConfig,
                                        fuse, fuse_batch, validate_anomaly, validate_threat)

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"agent", "schema_version", "record_id", "final_class", "decision", "evidence_status",
        "needs_review", "rule", "rationale", "evidence_issues", "signals"}


def a1(cls="BENIGN", conf=0.999, second="PORT_SCAN"):
    probs = {c: 0.0 for c in CLASSES}
    probs[cls] = conf
    other = second if second != cls else "BENIGN"
    probs[other] = round(1 - conf, 6)
    return {"agent": "threat_detection", "predicted_class": cls, "confidence": conf, "probabilities": probs}


def a2(score=0.5, threshold=0.97, reliable=True, rid="R1"):
    return {"record_id": rid, "anomaly_score": score, "is_anomaly": bool(score >= threshold),
            "threshold": threshold, "evidence": [{"feature": "Flow Duration"}, {"feature": "Fwd IAT Max"}],
            "data_quality": {"reliable": reliable, "imputed_model_features": [] if reliable else ["x"] * 9},
            "agent": "anomaly_detection"}


def check(r):
    assert set(r) == KEYS, "schema drift"
    assert r["decision"] in (ALLOW, BLOCK)
    assert r["final_class"] in (*CLASSES, UNKNOWN)
    assert r["needs_review"] == (r["evidence_status"] != "AGREEMENT")
    assert r["rationale"] and "attack probability of" not in r["rationale"]
    return r


# 1, 3 normal benign / agreement
def test_benign_agreement_allows():
    r = check(fuse(a1(), a2(0.4)))
    assert (r["final_class"], r["decision"], r["evidence_status"], r["needs_review"]) == \
        ("BENIGN", ALLOW, "AGREEMENT", False)


# 2, 3 every attack class + anomaly -> BLOCK with Agent 1's class
@pytest.mark.parametrize("cls", ATTACKS)
def test_attack_agreement_blocks(cls):
    r = check(fuse(a1(cls, 0.998), a2(0.995)))
    assert (r["final_class"], r["decision"], r["evidence_status"]) == (cls, BLOCK, "AGREEMENT")


# 4 disagreement, both directions; 7 low anomaly score never downgrades an attack
@pytest.mark.parametrize("cls", ATTACKS)
def test_attack_with_normal_score_still_blocks(cls):
    r = check(fuse(a1(cls, 0.999), a2(0.0)))
    assert (r["final_class"], r["decision"], r["evidence_status"]) == (cls, BLOCK, "DISAGREEMENT")


# 6 high anomaly score on confident BENIGN -> approved policy ALLOW + review
def test_confident_benign_with_anomaly_allows_with_review():
    r = check(fuse(a1("BENIGN", 0.9999), a2(1.0)))
    assert (r["final_class"], r["decision"], r["evidence_status"], r["needs_review"]) == \
        ("BENIGN", ALLOW, "DISAGREEMENT", True)
    assert "not an attack probability" in r["rationale"]


def test_benign_anomaly_policy_is_configurable():
    r = fuse(a1(), a2(0.99), FusionConfig(benign_anomaly_decision=BLOCK))
    assert r["decision"] == BLOCK and r["final_class"] == "BENIGN"


# 5 low confidence
def test_low_confidence_benign_blocks_as_uncertain():
    for score in (0.1, 0.99):
        r = check(fuse(a1("BENIGN", 0.75), a2(score)))
        assert (r["final_class"], r["decision"], r["evidence_status"]) == ("BENIGN", BLOCK, "UNCERTAIN")


def test_low_confidence_attack_blocks():
    r = check(fuse(a1("DOS", 0.6, second="DDOS"), a2(0.2)))
    assert (r["final_class"], r["decision"], r["evidence_status"]) == ("DOS", BLOCK, "UNCERTAIN")


def test_confidence_boundary():
    assert fuse(a1("BENIGN", 0.99), a2(0.1))["evidence_status"] == "AGREEMENT"   # >= floor
    assert fuse(a1("BENIGN", 0.989999), a2(0.1))["evidence_status"] == "UNCERTAIN"


def test_anomaly_threshold_boundary():
    assert fuse(a1(), a2(0.97))["rule"] == "benign_over_anomaly"                 # score == threshold
    assert fuse(a1(), a2(0.969999))["rule"] == "benign_agreement"


# 13 unreliable Agent 2 evidence (both uncertain when combined with low confidence)
@pytest.mark.parametrize("cls,conf", [("BENIGN", 0.9999), ("BENIGN", 0.6), ("DDOS", 0.999)])
def test_unreliable_anomaly_blocks(cls, conf):
    r = check(fuse(a1(cls, conf), a2(0.1, reliable=False)))
    assert (r["decision"], r["evidence_status"], r["needs_review"]) == (BLOCK, "UNCERTAIN", True)
    assert r["final_class"] == cls


# 8-17, 20: every malformed / missing / failed output -> INVALID_EVIDENCE + BLOCK, never ALLOW
def _bad_a1():
    nan, inf = float("nan"), float("inf")
    good = a1()
    cases = {
        "none": None, "empty": {}, "list": [0.9], "string": "BENIGN",
        "raised": RuntimeError("model file missing"),
        "unknown_class": {**good, "predicted_class": "MALWARE"},
        "lowercase_class": {**good, "predicted_class": "benign"},
        "class_not_str": {**good, "predicted_class": 0},
        "missing_conf": {k: v for k, v in good.items() if k != "confidence"},
        "nan_conf": {**good, "confidence": nan}, "inf_conf": {**good, "confidence": inf},
        "str_conf": {**good, "confidence": "0.999"}, "bool_conf": {**good, "confidence": True},
        "conf_mismatch": {**good, "confidence": 0.5},
        "probs_missing": {k: v for k, v in good.items() if k != "probabilities"},
        "probs_list": {**good, "probabilities": [0.999, 0.001, 0, 0, 0]},
        "probs_missing_class": {**good, "probabilities": {c: v for c, v in good["probabilities"].items() if c != "DOS"}},
        "probs_extra_class": {**good, "probabilities": {**good["probabilities"], "MALWARE": 0.0}},
        "prob_nan": {**good, "probabilities": {**good["probabilities"], "DOS": nan}},
        "prob_inf": {**good, "probabilities": {**good["probabilities"], "DOS": inf}},
        "prob_negative": {**good, "probabilities": {**good["probabilities"], "DOS": -0.1, "DDOS": 0.1}},
        "prob_above_one": {**good, "probabilities": {**good["probabilities"], "BENIGN": 1.5}},
        "prob_sum_low": {**good, "probabilities": {**good["probabilities"], "PORT_SCAN": 0.0, "BENIGN": 0.5},
                         "confidence": 0.5},
        "prob_sum_high": {**good, "probabilities": {c: 0.999 for c in CLASSES}},
        "class_not_argmax": {**good, "predicted_class": "DOS"},
    }
    return cases


@pytest.mark.parametrize("name,bad", list(_bad_a1().items()))
def test_invalid_agent1_blocks_unknown(name, bad):
    assert validate_threat(bad)
    r = check(fuse(bad, a2(0.1)))
    assert (r["final_class"], r["decision"], r["evidence_status"], r["needs_review"]) == \
        (UNKNOWN, BLOCK, "INVALID_EVIDENCE", True), name
    assert r["evidence_issues"] and r["signals"]["threat_class"] is None


def _bad_a2():
    nan, inf, good = float("nan"), float("inf"), a2(0.1)
    return {
        "none": None, "empty": {}, "list": [0.1], "raised": ValueError("no usable model features"),
        "missing_score": {k: v for k, v in good.items() if k != "anomaly_score"},
        "nan_score": {**good, "anomaly_score": nan}, "inf_score": {**good, "anomaly_score": inf},
        "neg_inf_score": {**good, "anomaly_score": -inf},
        "score_above_one": {**good, "anomaly_score": 1.5, "is_anomaly": True},
        "score_negative": {**good, "anomaly_score": -0.2},
        "score_str": {**good, "anomaly_score": "0.1"}, "score_bool": {**good, "anomaly_score": False},
        "flag_not_bool": {**good, "is_anomaly": 0}, "flag_contradicts": {**good, "is_anomaly": True},
        "missing_threshold": {k: v for k, v in good.items() if k != "threshold"},
        "nan_threshold": {**good, "threshold": nan},
        "missing_quality": {k: v for k, v in good.items() if k != "data_quality"},
        "reliable_not_bool": {**good, "data_quality": {"reliable": "yes"}},
        "evidence_not_list": {**good, "evidence": "Flow Duration"},
        "record_mismatch": {**good, "record_id": "OTHER"},
    }


@pytest.mark.parametrize("name,bad", list(_bad_a2().items()))
@pytest.mark.parametrize("cls", ["BENIGN", "DDOS"])
def test_invalid_agent2_blocks_keeps_agent1_class(name, bad, cls):
    assert validate_anomaly(bad, "R1")
    r = check(fuse(a1(cls, 0.9999), bad, record_id="R1"))
    assert (r["final_class"], r["decision"], r["evidence_status"]) == (cls, BLOCK, "INVALID_EVIDENCE"), name
    assert r["signals"]["anomaly_score"] is None   # never a fabricated or substituted score


def test_both_invalid():
    r = check(fuse(None, float("nan")))
    assert (r["final_class"], r["decision"], r["evidence_status"]) == (UNKNOWN, BLOCK, "INVALID_EVIDENCE")
    assert len(r["evidence_issues"]) == 2


def test_numpy_scalars_accepted():
    np = pytest.importorskip("numpy")
    x = a1()
    x["probabilities"] = {c: np.float32(v) for c, v in x["probabilities"].items()}
    x["confidence"] = np.float64(x["confidence"])
    assert fuse(x, a2(np.float64(0.2)))["decision"] == ALLOW


# Agent 2 never invents a class; the Bayes rule is off by default and only picks Agent 1 classes
def test_bayes_escalation_off_by_default():
    r = fuse(a1("BENIGN", 0.9429, second="PORT_SCAN"), a2(0.9993))
    assert r["final_class"] == "BENIGN" and r["rule"] == "low_confidence"
    r = fuse(a1("BENIGN", 0.995, second="DOS"), a2(0.9993))
    assert (r["final_class"], r["rule"]) == ("BENIGN", "benign_over_anomaly")


def test_bayes_escalation_when_enabled_uses_agent1_attack_class():
    cfg = FusionConfig(bayes_escalation=True)
    r = check(fuse(a1("BENIGN", 0.995, second="DOS"), a2(0.9993), cfg))
    # odds = 0.005/0.995 * 32.5 = 0.163 -> posterior 0.14 < 0.5: stays BENIGN
    assert r["final_class"] == "BENIGN"
    r = check(fuse(a1("BENIGN", 0.9, second="DOS"), a2(0.9993), FusionConfig(bayes_escalation=True, min_confidence=0.8)))
    assert (r["final_class"], r["decision"], r["rule"]) == ("DOS", BLOCK, "bayes_escalation")
    r = fuse(a1("BENIGN", 0.9, second="DOS"), a2(0.5), FusionConfig(bayes_escalation=True, min_confidence=0.8))
    assert r["final_class"] == "BENIGN"   # no anomaly flag -> no escalation


def test_final_class_never_from_agent2():
    x = a2(0.9999)
    x["predicted_class"] = "DDOS"           # a stray class field on Agent 2 output is ignored
    assert fuse(a1("BENIGN", 0.9999), x)["final_class"] == "BENIGN"


def test_config_validation():
    for bad in ({"benign_anomaly_decision": "INVESTIGATE"}, {"invalid_decision": "allow"},
                {"min_confidence": 0}, {"min_confidence": 1.5}, {"a2_likelihood_ratio": math.inf},
                {"a2_likelihood_ratio": 0}):
        with pytest.raises(ValueError):
            FusionConfig(**bad)


# 17-19 empty inputs, batches, repeated identical requests, no input mutation
def test_batch_matches_single_and_is_deterministic():
    xs = [a1(), a1("DDOS", 0.999), a1("BENIGN", 0.7), None]
    ys = [a2(0.1, rid="A"), a2(0.99, rid="B"), a2(0.2, rid="C"), a2(0.3, rid="D")]
    ids = ["A", "B", "C", "D"]
    before = copy.deepcopy((xs, ys))
    batch = fuse_batch(xs, ys, ids)
    assert batch == [fuse(x, y, record_id=i) for x, y, i in zip(xs, ys, ids)]
    assert batch == fuse_batch(xs, ys, ids)
    assert (xs, ys) == before, "inputs mutated"
    assert [r["record_id"] for r in batch] == ids
    assert fuse_batch([], []) == []


def test_batch_misalignment_raises():
    with pytest.raises(ValueError):
        fuse_batch([a1()], [a2(), a2()])
    with pytest.raises(ValueError):
        fuse_batch([a1()], [a2()], ["A", "B"])


# ---------------------------------------------------------------- integration (real models)
MODEL = ROOT / "models" / "anomaly_detector.joblib"
needs_models = pytest.mark.skipif(not MODEL.exists(), reason="Agent 2 model not trained")


@pytest.fixture(scope="module")
def sample():
    pd = pytest.importorskip("pandas")
    ev = pd.read_parquet(ROOT / "data" / "raw" / "eval" / "flows.parquet")
    return ev.groupby("label", group_keys=False).head(20).reset_index(drop=True)


@needs_models
def test_end_to_end_real_agents(sample):
    from run_pipeline import run
    a1s, a2s, fused, _ = run(sample)
    assert len(fused) == len(sample)
    for r, rid in zip(fused, sample.record_id):
        check(r)
        assert r["record_id"] == rid and r["evidence_status"] != "INVALID_EVIDENCE"
    assert run(sample)[2] == fused                          # repeated identical requests
    for (_, row), r in zip(sample.head(5).iterrows(), fused):
        from agents.threat_detection import predict_threat
        from src.agents.anomaly_detection import detect_anomaly
        rec = row.to_dict()
        assert fuse(predict_threat(rec), detect_anomaly(rec), record_id=rec["record_id"]) == r


@needs_models
def test_end_to_end_dirty_rows_never_allow_silently(sample):
    import numpy as np
    from run_pipeline import run
    dirty = sample.head(4).copy()
    feats = [c for c in dirty.columns if c not in ("record_id", "label")]
    dirty.loc[0, feats[:40]] = np.nan                        # heavily missing -> Agent 2 unreliable
    dirty.loc[1, "Flow Bytes/s"] = np.inf
    dirty.loc[2, feats] = np.nan                             # nothing usable -> Agent 2 raises
    r = run(dirty)[2]
    assert r[0]["decision"] == BLOCK and r[0]["evidence_status"] in ("UNCERTAIN", "INVALID_EVIDENCE")
    assert r[2]["decision"] == BLOCK and r[2]["evidence_status"] == "INVALID_EVIDENCE"
    assert all(check(x) for x in r)


@needs_models
def test_end_to_end_agent_failure(sample, monkeypatch):
    import run_pipeline

    def boom(df):
        raise RuntimeError("anomaly service down")
    monkeypatch.setattr(run_pipeline, "detect_anomalies", boom)
    r = run_pipeline.run(sample.head(3))[2]
    assert all(x["decision"] == BLOCK and x["evidence_status"] == "INVALID_EVIDENCE" for x in r)
    assert all("anomaly service down" in x["evidence_issues"][0] for x in r)
