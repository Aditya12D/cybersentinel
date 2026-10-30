import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from src.agents.anomaly_detection import AnomalyDetectionAgent, config
from src.agents.anomaly_detection.detector import AnomalyDetector


def test_scores_in_unit_interval_and_benign_roughly_uniform(detector, benign_ref):
    s = detector.anomaly_score(benign_ref)
    assert s.min() >= 0 and s.max() <= 1
    # by construction ~3% of the reference set lies at/above the 0.97 percentile
    assert abs(np.mean(s >= 0.97) - 0.03) < 0.01


def test_attacks_score_higher_than_benign(detector, attacks, benign_ref):
    sa, sb = detector.anomaly_score(attacks), detector.anomaly_score(benign_ref)
    assert np.median(sa) > 0.99 > np.median(sb)
    assert detector.predict(attacks).mean() > 0.95


def test_batch_equals_single_bitwise(detector, attacks, benign_ref):
    df = pd.concat([attacks.iloc[:20], benign_ref.iloc[:20]])
    batch = detector.score(df)
    for i, r in enumerate(df.to_dict("records")):
        one = detector.score([r])
        assert one.anomaly_score[0] == batch.anomaly_score[i]
        assert one.knn_distance[0] == batch.knn_distance[i]
        assert one.iforest_percentile[0] == batch.iforest_percentile[i]


def test_kd_tree_and_brute_paths_agree(detector, benign_ref):
    from src.agents.anomaly_detection import detector as dmod
    Z = detector.preprocessor.transform(benign_ref.iloc[:600])[:, detector.model_idx_]
    d_batch, _ = detector._knn(Z)                      # brute path (600 > KD_TREE_MAX_ROWS)
    step = dmod.KD_TREE_MAX_ROWS
    d_small = np.concatenate([detector._knn(Z[i:i + step])[0] for i in range(0, 600, step)])
    np.testing.assert_array_equal(d_batch, d_small)


def test_derived_caches_not_pickled_but_rebuilt(detector, tmp_path):
    state = detector.__getstate__()
    assert not set(AnomalyDetector._DERIVED) & set(state)
    p = tmp_path / "d.joblib"
    detector.save(p)
    loaded = AnomalyDetector.load(p)
    assert loaded.knn_tree_ is not None and loaded.iforest_fast_ is not None
    assert loaded.threshold == detector.threshold


def test_reproducible_with_fixed_seed(benign_fit, benign_ref, attacks):
    a = AnomalyDetector(seed=7).fit(benign_fit, benign_ref).anomaly_score(attacks)
    b = AnomalyDetector(seed=7).fit(benign_fit, benign_ref).anomaly_score(attacks)
    np.testing.assert_array_equal(a, b)


def test_save_load_roundtrip(detector, attacks, tmp_path):
    p = tmp_path / "m.joblib"
    detector.save(p)
    np.testing.assert_array_equal(AnomalyDetector.load(p).anomaly_score(attacks),
                                  detector.anomaly_score(attacks))


SCHEMA_KEYS = {"record_id", "anomaly_score", "is_anomaly", "evidence", "reason", "rationale",
               "threshold", "components", "data_quality", "agent", "schema_version",
               "model_version"}


def test_agent_output_schema_and_json(detector, attacks, benign_ref):
    agent = AnomalyDetectionAgent(detector=detector)
    flows = list(attacks.iloc[:5].to_dict("records")) + list(benign_ref.iloc[:5].to_dict("records"))
    results = agent.detect_anomalies(flows)
    for r in results:
        assert set(r) == SCHEMA_KEYS
        assert isinstance(r["reason"], str) and r["reason"] == r["rationale"]
        assert 0.0 <= r["anomaly_score"] <= 1.0
        assert set(r["data_quality"]) == {"missing_features", "invalid_features",
                                          "non_finite_features", "imputed_model_features",
                                          "reliable"}
        assert isinstance(r["is_anomaly"], bool)
        assert r["is_anomaly"] == (r["anomaly_score"] >= r["threshold"])
    json.dumps(results)
    assert results[0]["record_id"] == attacks.iloc[0]["record_id"]
    single = agent.detect_anomaly(flows[0])
    assert single["anomaly_score"] == results[0]["anomaly_score"]


def test_agent_ignores_extra_fields_and_label(detector, attacks):
    agent = AnomalyDetectionAgent(detector=detector)
    flow = attacks.iloc[0].to_dict()
    base = agent.detect_anomaly(flow)["anomaly_score"]
    flow.update({"Source IP": "10.0.0.1", "Timestamp": "x", "label": "BENIGN",
                 "agent1_prediction": "BENIGN"})
    assert agent.detect_anomaly(flow)["anomaly_score"] == base


def test_empty_batch(detector):
    assert AnomalyDetectionAgent(detector=detector).detect_anomalies([]) == []


@pytest.mark.skipif(not config.MODEL_PATH.exists() or not config.EVAL_PATH.exists(),
                    reason="trained model / eval data not available")
def test_trained_model_on_real_eval_sample():
    import pandas as pd
    ev = pd.read_parquet(config.EVAL_PATH)
    sample = pd.concat([g.sample(200, random_state=0) for _, g in ev.groupby("label")])
    agent = AnomalyDetectionAgent()
    res = agent.detect_anomalies(sample.drop(columns="label"))
    flagged = np.array([r["is_anomaly"] for r in res])
    is_attack = (sample["label"] != "BENIGN").to_numpy()
    assert flagged[is_attack].mean() > 0.9
    assert flagged[~is_attack].mean() < 0.1


def test_no_decision_or_attack_class_in_output(detector, attacks):
    r = AnomalyDetectionAgent(detector=detector).detect_anomaly(attacks.iloc[0].to_dict())
    text = json.dumps(r).upper()
    for forbidden in ("ALLOW", "BLOCK", "PORT_SCAN", "DDOS", "BRUTE_FORCE"):
        assert forbidden not in text
    assert not {"decision", "action", "predicted_class", "label"} & set(r)


def test_input_validation(detector, attacks):
    agent = AnomalyDetectionAgent(detector=detector)
    with pytest.raises(TypeError):
        agent.detect_anomaly("flow")
    with pytest.raises(TypeError):
        agent.detect_anomalies(["flow"])
    with pytest.raises(ValueError, match="no usable model features"):
        agent.detect_anomaly({"record_id": "x", "Source IP": "10.0.0.1"})
    with pytest.raises(ValueError):
        AnomalyDetectionAgent(detector=detector, threshold=1.5)
    # partially missing input: still scored, marked unreliable, reported
    flow = {k: v for k, v in attacks.iloc[0].to_dict().items() if k in ("Flow Duration", "record_id")}
    r = agent.detect_anomaly(flow)
    assert r["data_quality"]["reliable"] is False
    assert r["data_quality"]["missing_features"]
    assert agent.detect_anomaly({"Flow Duration": 5.0})["record_id"] is None


def test_series_input_and_dataframe_order(detector, attacks, benign_ref):
    agent = AnomalyDetectionAgent(detector=detector)
    df = pd.concat([benign_ref.iloc[:3], attacks.iloc[:3]])
    res = agent.detect_anomalies(df)
    assert [r["record_id"] for r in res] == df["record_id"].tolist()
    assert agent.detect_anomaly(df.iloc[4])["anomaly_score"] == res[4]["anomaly_score"]


def test_thread_safe_concurrent_calls(detector, attacks, benign_ref):
    agent = AnomalyDetectionAgent(detector=detector)
    flows = list(pd.concat([attacks.iloc[:20], benign_ref.iloc[:20]]).to_dict("records"))
    expected = [agent.detect_anomaly(f) for f in flows]
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(agent.detect_anomaly, flows))
    assert got == expected


def test_public_import_path():
    from src.agents.anomaly_detection import detect_anomalies, detect_anomaly  # noqa: F401
    assert callable(detect_anomaly) and callable(detect_anomalies)
