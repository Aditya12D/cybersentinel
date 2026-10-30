import numpy as np

from src.agents.anomaly_detection import AnomalyDetectionAgent, config
from src.agents.anomaly_detection.evidence import (build_rationale, compute_evidence,
                                                   feature_category, severity_from_tail)


def test_severity_levels_follow_tail_probability():
    assert severity_from_tail(0.0005) == "high"
    assert severity_from_tail(0.005) == "medium"
    assert severity_from_tail(0.03) == "low"
    assert severity_from_tail(0.3) == "info"


def test_feature_categories():
    assert feature_category("Flow Packets/s") == "packet rate"
    assert feature_category("Flow IAT Mean") == "packet inter-arrival time"
    assert feature_category("Fwd Packet Length Mean") == "packet size"
    assert feature_category("RST Flag Count") == "TCP flag usage"


def test_evidence_is_grounded_in_observed_values(detector, attacks):
    res = detector.score(attacks)
    flags = res.anomaly_score >= detector.threshold
    ev = compute_evidence(detector, res, flags)
    feats = set(detector.model_features)
    for i in np.where(flags)[0][:50]:
        assert 1 <= len(ev[i]) <= config.EVIDENCE_TOP_K
        for item in ev[i]:
            assert item["feature"] in feats
            # observed value equals the (imputed) input value for that feature
            assert item["observed_value"] == float(res.raw[i, detector.features.index(item["feature"])])
            assert 0 <= item["distance_contribution"] <= 1
            assert item["severity"] in {"high", "medium", "low", "info"}
            assert item["severity"] == severity_from_tail(item["benign_tail_probability"])
            lo, hi = item["nearest_benign_range"]
            assert lo <= hi


def test_port_scan_like_flow_reports_rate_or_duration(detector, attacks):
    agent = AnomalyDetectionAgent(detector=detector)
    r = agent.detect_anomaly(attacks.iloc[0].to_dict())
    assert r["is_anomaly"]
    top = {e["feature"] for e in r["evidence"]}
    assert top & {"Flow Packets/s", "Flow Duration", "Fwd Packet Length Mean", "RST Flag Count"}
    high = [e for e in r["evidence"] if e["feature"] == "Flow Packets/s"]
    if high:
        assert high[0]["direction"] == "above"
        assert "Above benign reference range" in high[0]["reference"]


def test_normal_flows_only_report_rare_values(detector, benign_ref):
    res = detector.score(benign_ref)
    flags = np.zeros(len(benign_ref), dtype=bool)
    for items in compute_evidence(detector, res, flags):
        assert all(e["severity"] in {"high", "medium"} for e in items)


def test_rationale_mentions_score_threshold_and_evidence():
    ev = [{"feature": "Flow Packets/s", "severity": "high", "direction": "above"}]
    txt = build_rationale(0.995, 0.97, True, 0.999, 0.5, ev)
    assert "0.995" in txt and "0.970" in txt and "packet rate" in txt
    assert "not" not in build_rationale(0.995, 0.97, True, 0.999, 0.5, ev).split(".")[0]
    assert "consistent with normal" in build_rationale(0.2, 0.97, False, 0.1, 0.3, [])


def test_imputed_features_never_reported_as_evidence(detector, attacks):
    agent = AnomalyDetectionAgent(detector=detector)
    flow = attacks.iloc[0].to_dict()
    clean = agent.detect_anomaly(flow)
    top = clean["evidence"][0]["feature"]
    flow[top] = float("inf")
    flow["Flow Duration"] = "garbage"
    r = agent.detect_anomaly(flow)
    feats = {e["feature"] for e in r["evidence"]}
    assert top not in feats and "Flow Duration" not in feats
    assert top in r["data_quality"]["non_finite_features"]
    assert "Flow Duration" in r["data_quality"]["invalid_features"]
    assert set(r["data_quality"]["imputed_model_features"]) >= {top, "Flow Duration"}
    assert "replaced by benign reference values" in r["reason"]


def test_evidence_values_match_input_exactly(detector, attacks):
    agent = AnomalyDetectionAgent(detector=detector)
    flow = attacks.iloc[1].to_dict()
    for e in agent.detect_anomaly(flow)["evidence"]:
        assert e["observed_value"] == float(flow[e["feature"]])
