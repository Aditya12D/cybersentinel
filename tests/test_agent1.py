"""Agent 1 contract check: python -m pytest tests -q (needs models/ from `train_agent1.py final`)."""
import math

import pandas as pd

from agents.threat_detection import CLASSES, ROOT, predict_batch, predict_threat

EV = pd.read_parquet(ROOT / "data" / "raw" / "eval" / "flows.parquet")


def test_contract_on_real_rows():
    for label in CLASSES:
        row = EV[EV.label == label].iloc[0].drop(["label"]).to_dict()  # record_id ignored
        out = predict_threat(row)
        assert out["agent"] == "threat_detection"
        assert list(out["probabilities"]) == CLASSES
        assert math.isclose(sum(out["probabilities"].values()), 1, abs_tol=1e-4)
        assert out["confidence"] == max(out["probabilities"].values())
        assert out["predicted_class"] == max(out["probabilities"], key=out["probabilities"].get)
        assert out["predicted_class"] == label


def test_dirty_input_does_not_crash():
    row = EV.iloc[0].drop(["label", "record_id"]).to_dict()
    row["Flow Bytes/s"] = float("inf")
    row["Flow Packets/s"] = float("-inf")
    row["Fwd Packets/s"] = None
    row["Bwd Packets/s"] = "not-a-number"
    del row["Destination Port"]
    out = predict_threat(row)
    assert out["predicted_class"] in CLASSES


def test_batch_matches_single():
    df = EV.head(50)
    batch = predict_batch(df)
    assert batch[7] == predict_threat(df.iloc[7].to_dict())
