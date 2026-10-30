"""Shared fixtures: a small synthetic flow dataset so unit tests run in seconds
without the real CICIDS2017 files or a trained model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.anomaly_detection.detector import AnomalyDetector  # noqa: E402

FEATURES = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Flow Packets/s",
            "Fwd Packet Length Mean", "Avg Fwd Segment Size", "Fwd URG Flags",
            "RST Flag Count", "Init_Win_bytes_forward", "Fwd Header Length"]


def make_flows(n: int, seed: int, attack: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = {
        "Destination Port": rng.choice([53, 80, 443], n).astype(float),
        "Flow Duration": rng.lognormal(10, 1.5, n),
        "Total Fwd Packets": rng.poisson(5, n).astype(float) + 1,
        "Flow Packets/s": rng.lognormal(4, 1, n),
        "Fwd Packet Length Mean": rng.lognormal(4, 0.5, n),
        "Fwd URG Flags": np.zeros(n),
        "RST Flag Count": rng.binomial(1, 0.1, n).astype(float),
        "Init_Win_bytes_forward": rng.choice([0, 8192, 65535], n).astype(float),
        "Fwd Header Length": rng.normal(100, 20, n),
    }
    if attack:  # port-scan-like: tiny, very fast flows
        d["Flow Duration"] = rng.uniform(20, 60, n)
        d["Flow Packets/s"] = rng.uniform(3e4, 6e4, n)
        d["Fwd Packet Length Mean"] = np.zeros(n)
        d["RST Flag Count"] = np.full(n, 3.0)
    d["Avg Fwd Segment Size"] = d["Fwd Packet Length Mean"]  # exact duplicate column
    df = pd.DataFrame(d)[FEATURES].astype(np.float32)
    df.insert(0, "record_id", [f"T{seed}_{i}" for i in range(n)])
    df["label"] = "PORT_SCAN" if attack else "BENIGN"
    return df


@pytest.fixture(scope="session")
def benign_fit():
    return make_flows(3000, 1)


@pytest.fixture(scope="session")
def benign_ref():
    return make_flows(1000, 2)


@pytest.fixture(scope="session")
def attacks():
    return make_flows(200, 3, attack=True)


@pytest.fixture(scope="session")
def detector(benign_fit, benign_ref):
    det = AnomalyDetector().fit(benign_fit, benign_ref)
    det.threshold = 0.97
    return det
