import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from src.agents.anomaly_detection.fast_iforest import CompiledIsolationForest


@pytest.fixture(scope="module")
def forest_and_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 12)).astype(np.float32)
    # many exact ties, as in real benign flows
    X[:1000] = X[0]
    f = IsolationForest(n_estimators=100, max_samples=1024, random_state=0).fit(X)
    Q = np.vstack([X[:500], rng.normal(scale=3, size=(500, 12)).astype(np.float32)])
    return f, Q


def test_identical_to_sklearn(forest_and_data):
    f, Q = forest_and_data
    np.testing.assert_array_equal(CompiledIsolationForest(f).score_samples(Q), f.score_samples(Q))


def test_single_row_equals_batch_bitwise(forest_and_data):
    f, Q = forest_and_data
    c = CompiledIsolationForest(f)
    batch = c.score_samples(Q[:50])
    single = np.array([c.score_samples(Q[i:i + 1])[0] for i in range(50)])
    np.testing.assert_array_equal(batch, single)


def test_rejects_feature_subsampling():
    X = np.random.default_rng(0).normal(size=(500, 6)).astype(np.float32)
    f = IsolationForest(n_estimators=10, max_features=0.5, random_state=0).fit(X)
    with pytest.raises(ValueError):
        CompiledIsolationForest(f)


def test_shape_validation(forest_and_data):
    f, _ = forest_and_data
    with pytest.raises(ValueError):
        CompiledIsolationForest(f).score_samples(np.zeros((2, 3), dtype=np.float32))
