"""Reproducible preprocessing for Agent 2.

Pipeline (all statistics fitted on BENIGN *training* flows only):

1. Column selection
   * ``record_id`` and ``label`` are metadata and never reach the model.
   * Forced exclusions (``config.FORCED_EXCLUSIONS``, e.g. Destination Port).
   * Constant / near-constant columns in the fit data (no information).
   * Exact duplicate columns (keep the first occurrence in original order).
   Feature order of the surviving columns follows the original file order.

2. Non-finite handling (none present in the released data, but the replay data
   may contain them, as raw CICFlowMeter output often does for rate features)
   * NaN  -> BENIGN median of that feature (neutral, adds no anomaly signal).
   * +inf -> maximum observed in the fit data; -inf -> minimum. An infinite
     rate is genuinely extreme, so it is mapped to the extreme of the known
     range rather than hidden behind the median.
   Every imputation is reported back so the caller can surface it.

3. Signed log transform ``sign(x) * log1p(|x|)``
   Most volume/rate/IAT features span 6-9 orders of magnitude with skew up to
   ~390. The log compresses that range so that both tree split points and
   distance-based scores are not governed by a handful of huge values. The
   signed variant keeps the (artefactual) negative header lengths ordered.

4. Robust standardisation ``z = (t - median) / scale`` in log space
   ``scale`` = IQR / 1.349 (a robust sigma); if the IQR is zero (common for
   zero-inflated features) the standard deviation is used instead, and 1.0 as a
   last resort. z is winsorised to +/- ``config.ROBUST_Z_CLIP``.

Raw values are never modified in place; metadata is returned separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from . import config


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class ImputationReport:
    """Per-row record of which features had to be repaired.

    missing     - column absent from the input, or value null/None/NaN
    invalid     - value present but not numeric (e.g. "abc")
    non_finite  - +inf / -inf
    imputed     - boolean matrix (n, n_features), True where the model value
                  is a repair rather than an observed value
    """

    missing: list[list[str]] = field(default_factory=list)
    invalid: list[list[str]] = field(default_factory=list)
    non_finite: list[list[str]] = field(default_factory=list)
    imputed: np.ndarray | None = None


class FlowPreprocessor:
    """Fit-on-benign, transform-anything preprocessor."""

    def __init__(self) -> None:
        self.input_features_: list[str] | None = None
        self.features_: list[str] | None = None
        self.dropped_: dict[str, str] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X_benign: pd.DataFrame) -> "FlowPreprocessor":
        df = X_benign.drop(columns=[c for c in config.METADATA_COLUMNS if c in X_benign.columns])
        self.input_features_ = list(df.columns)
        dropped: dict[str, str] = {}

        for col, reason in config.FORCED_EXCLUSIONS.items():
            if col in df.columns:
                dropped[col] = reason

        values = df.to_numpy(dtype=np.float64)
        n = len(df)
        for j, col in enumerate(df.columns):
            if col in dropped:
                continue
            v = values[:, j]
            uniq, counts = np.unique(v[np.isfinite(v)], return_counts=True)
            if len(uniq) <= 1:
                dropped[col] = "constant in benign training data"
            elif n - counts.max() < config.NEAR_CONSTANT_MAX_OTHER_ROWS:
                dropped[col] = f"near-constant ({n - counts.max()} non-mode rows)"

        kept = [c for c in df.columns if c not in dropped]
        # Exact duplicate columns: compare on the fit data, keep first in order.
        seen: dict[bytes, str] = {}
        for col in kept:
            key = np.ascontiguousarray(df[col].to_numpy(dtype=np.float64)).tobytes()
            if key in seen:
                dropped[col] = f"exact duplicate of '{seen[key]}'"
            else:
                seen[key] = col

        self.features_ = [c for c in df.columns if c not in dropped]
        self.dropped_ = dropped

        raw = df[self.features_].to_numpy(dtype=np.float64)
        finite = np.where(np.isfinite(raw), raw, np.nan)
        self.raw_median_ = np.nanmedian(finite, axis=0)
        self.raw_min_ = np.nanmin(finite, axis=0)
        self.raw_max_ = np.nanmax(finite, axis=0)

        t = signed_log1p(np.where(np.isnan(finite), self.raw_median_, finite))
        self.log_median_ = np.median(t, axis=0)
        q25, q75 = np.percentile(t, [25, 75], axis=0)
        scale = (q75 - q25) / 1.349
        std = t.std(axis=0)
        scale = np.where(scale > 0, scale, std)
        self.log_scale_ = np.where(scale > 0, scale, 1.0)
        return self

    # ------------------------------------------------------------ transform
    def _check_fitted(self) -> None:
        if self.features_ is None:
            raise RuntimeError("FlowPreprocessor is not fitted")

    def to_raw_matrix(self, X: pd.DataFrame | Mapping | Iterable[Mapping],
                      report: ImputationReport | None = None) -> np.ndarray:
        """Select model features (in fitted order) and repair missing/invalid/
        non-finite values. Returns raw-unit float64 values, shape (n, n_features).

        Column names are matched exactly first, then after stripping
        surrounding whitespace (raw CICFlowMeter CSV headers carry a leading
        space, e.g. " Flow Duration")."""
        self._check_fitted()
        df = _as_frame(X)
        n = len(df)
        stripped = {}
        for c in df.columns:
            if isinstance(c, str):
                stripped.setdefault(c.strip(), c)
        out = np.empty((n, len(self.features_)), dtype=np.float64)
        missing = np.zeros_like(out, dtype=bool)
        invalid = np.zeros_like(out, dtype=bool)
        for j, col in enumerate(self.features_):
            src = col if col in df.columns else stripped.get(col)
            if src is None:
                out[:, j] = np.nan
                missing[:, j] = True
                continue
            values = df[src]
            if pd.api.types.is_numeric_dtype(values.dtype):  # fast path
                num = values.to_numpy(dtype=np.float64, na_value=np.nan)
                out[:, j] = num
                missing[:, j] = np.isnan(num)
                continue
            null = values.isna().to_numpy()
            num = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
            out[:, j] = num
            missing[:, j] = null
            invalid[:, j] = np.isnan(num) & ~null

        pinf, ninf = np.isposinf(out), np.isneginf(out)
        nan_mask = np.isnan(out)
        if report is not None:
            feats = np.array(self.features_)
            nonfin = pinf | ninf
            any_issue = (missing | invalid | nonfin).any(axis=1)
            for i in range(n):
                if not any_issue[i]:
                    report.missing.append([]); report.invalid.append([]); report.non_finite.append([])
                    continue
                report.missing.append(feats[missing[i]].tolist())
                report.invalid.append(feats[invalid[i]].tolist())
                report.non_finite.append(feats[nonfin[i]].tolist())
            report.imputed = nan_mask | pinf | ninf
        out = np.where(nan_mask, self.raw_median_, out)
        out = np.where(pinf, self.raw_max_, out)
        out = np.where(ninf, self.raw_min_, out)
        return out

    def raw_to_model(self, raw: np.ndarray) -> np.ndarray:
        """Raw-unit matrix -> standardised log-space model input (float32)."""
        z = (signed_log1p(raw) - self.log_median_) / self.log_scale_
        return np.clip(z, -config.ROBUST_Z_CLIP, config.ROBUST_Z_CLIP).astype(np.float32)

    def transform(self, X, report: ImputationReport | None = None) -> np.ndarray:
        return self.raw_to_model(self.to_raw_matrix(X, report))

    def describe(self) -> dict:
        self._check_fitted()
        return {
            "n_input_features": len(self.input_features_),
            "n_model_features": len(self.features_),
            "model_features": list(self.features_),
            "dropped_features": dict(self.dropped_),
        }


def _as_frame(X) -> pd.DataFrame:
    """Accept a DataFrame, a single mapping / Series, or an iterable of mappings."""
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, pd.Series):
        return X.to_frame().T
    if isinstance(X, Mapping):
        return pd.DataFrame([dict(X)])
    if isinstance(X, (str, bytes)) or not isinstance(X, Iterable):
        raise TypeError(f"expected a DataFrame, mapping or iterable of mappings, got {type(X).__name__}")
    rows = list(X)
    for i, r in enumerate(rows):
        if not isinstance(r, (Mapping, pd.Series)):
            raise TypeError(f"flow at position {i} is {type(r).__name__}, expected a mapping")
    return pd.DataFrame([dict(r) for r in rows])


def load_flows(path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a flows parquet file. Features stay float32 (as stored) to keep the
    full training set at ~190 MB in memory."""
    return pd.read_parquet(path, columns=columns)


def split_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features, metadata) without copying feature data unnecessarily."""
    meta_cols = [c for c in config.METADATA_COLUMNS if c in df.columns]
    return df.drop(columns=meta_cols), df[meta_cols]


def binary_target(labels: pd.Series) -> np.ndarray:
    """BENIGN -> 0, every attack class -> 1 (evaluation only)."""
    return (labels.to_numpy() != config.BENIGN_LABEL).astype(np.int8)
