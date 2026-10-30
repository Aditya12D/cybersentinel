import numpy as np
import pandas as pd
import pytest

from src.agents.anomaly_detection.datasets import benign_split
from src.agents.anomaly_detection.preprocessing import FlowPreprocessor, ImputationReport
from tests.anomaly_detection.conftest import make_flows


def test_metadata_and_bad_columns_excluded(benign_fit):
    pre = FlowPreprocessor().fit(benign_fit)
    assert "record_id" not in pre.features_ and "label" not in pre.features_
    assert "Destination Port" in pre.dropped_          # forced exclusion
    assert "Fwd URG Flags" in pre.dropped_             # constant
    assert "duplicate" in pre.dropped_["Avg Fwd Segment Size"]
    # surviving features keep original file order
    order = [c for c in benign_fit.columns if c in pre.features_]
    assert order == pre.features_


def test_transform_shape_dtype_and_column_order_independence(benign_fit):
    pre = FlowPreprocessor().fit(benign_fit)
    Z1 = pre.transform(benign_fit)
    shuffled = benign_fit[benign_fit.columns[::-1]]
    Z2 = pre.transform(shuffled)
    assert Z1.shape == (len(benign_fit), len(pre.features_))
    assert Z1.dtype == np.float32
    np.testing.assert_array_equal(Z1, Z2)
    # robust centring: benign medians map to ~0
    assert np.all(np.abs(np.median(Z1, axis=0)) < 0.05)


def test_nan_inf_missing_are_repaired_and_reported(benign_fit):
    pre = FlowPreprocessor().fit(benign_fit)
    row = benign_fit.iloc[0].to_dict()
    row["Flow Packets/s"] = float("inf")
    row["Flow Duration"] = float("nan")
    row["Fwd Header Length"] = float("-inf")
    del row["RST Flag Count"]
    rep = ImputationReport()
    raw = pre.to_raw_matrix([row], rep)
    assert np.isfinite(raw).all()
    j = pre.features_.index
    assert raw[0, j("Flow Packets/s")] == pre.raw_max_[j("Flow Packets/s")]
    assert raw[0, j("Fwd Header Length")] == pre.raw_min_[j("Fwd Header Length")]
    assert raw[0, j("Flow Duration")] == pre.raw_median_[j("Flow Duration")]
    # NaN / absent -> missing; +/-inf -> non_finite
    assert set(rep.missing[0]) == {"Flow Duration", "RST Flag Count"}
    assert set(rep.non_finite[0]) == {"Flow Packets/s", "Fwd Header Length"}
    assert rep.invalid[0] == []
    assert np.isfinite(pre.transform([row])).all()


def test_fit_only_uses_fit_data(benign_fit, attacks):
    a = FlowPreprocessor().fit(benign_fit)
    b = FlowPreprocessor().fit(benign_fit)
    b.transform(pd.concat([benign_fit, attacks]))  # transforming must not refit
    np.testing.assert_array_equal(a.log_median_, b.log_median_)
    np.testing.assert_array_equal(a.log_scale_, b.log_scale_)


def test_whitespace_in_column_names_is_tolerated(benign_fit):
    pre = FlowPreprocessor().fit(benign_fit)
    padded = benign_fit.rename(columns=lambda c: f" {c}" if c not in ("record_id", "label") else c)
    np.testing.assert_array_equal(pre.transform(padded), pre.transform(benign_fit))


def test_non_numeric_values_reported_as_invalid(benign_fit):
    pre = FlowPreprocessor().fit(benign_fit)
    row = benign_fit.iloc[0].to_dict()
    row["Flow Duration"] = "abc"
    row["Total Fwd Packets"] = None
    rep = ImputationReport()
    raw = pre.to_raw_matrix([row], rep)
    assert np.isfinite(raw).all()
    assert rep.invalid[0] == ["Flow Duration"]
    assert rep.missing[0] == ["Total Fwd Packets"]
    assert rep.imputed[0, pre.features_.index("Flow Duration")]
    assert not rep.imputed[0, pre.features_.index("Flow Packets/s")]


def test_series_and_bad_input_types(benign_fit):
    pre = FlowPreprocessor().fit(benign_fit)
    s = benign_fit.iloc[3]
    np.testing.assert_array_equal(pre.transform(s), pre.transform(benign_fit.iloc[[3]]))
    for bad in ("not a flow", 42, [1, 2, 3]):
        with pytest.raises(TypeError):
            pre.transform(bad)


def test_benign_split_deduplicates_and_is_disjoint():
    df = make_flows(400, 9)
    dup = df.iloc[:100].copy()
    dup["record_id"] = dup["record_id"] + "_dup"
    attack = make_flows(50, 10, attack=True)
    fit, ref = benign_split(pd.concat([df, dup, attack], ignore_index=True))
    assert set(fit.label) == {"BENIGN"} and set(ref.label) == {"BENIGN"}
    feats = [c for c in df.columns if c not in ("record_id", "label")]
    hf = set(pd.util.hash_pandas_object(fit[feats], index=False))
    assert not pd.util.hash_pandas_object(ref[feats], index=False).isin(hf).any()
    assert len(fit) + len(ref) == 400
