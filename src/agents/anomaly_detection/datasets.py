"""Deterministic data partitions used by training, calibration and evaluation.

* BENIGN training flows (exact feature-duplicates removed) are split into
    - ``fit``       (80%) - used to fit the preprocessor and the detector;
    - ``reference`` (20%) - never seen by the detector; used to map raw scores
      onto the normalised anomaly score (benign percentile) and to build the
      benign feature reference used for evidence.
  Attack rows from the training partition are NOT used by the detector at all.

* The labelled development eval partition is split, stratified by label, into
    - ``calibration`` (50%) - threshold selection and model selection;
    - ``holdout``     (50%) - reported once, after every choice was frozen.
  Metrics on the full eval partition are reported too, for comparability with
  the other agents, with the caveat that half of it informed the threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .preprocessing import load_flows


@dataclass
class Splits:
    benign_fit: pd.DataFrame
    benign_reference: pd.DataFrame
    eval_calibration: pd.DataFrame
    eval_holdout: pd.DataFrame


def benign_split(train: pd.DataFrame, seed: int = config.RANDOM_SEED):
    """BENIGN rows only, exact feature-duplicates removed *before* splitting.

    De-duplication guarantees the reference set contains no flow whose feature
    vector is also in the fit set. Without it ~6% of reference rows were
    memorised copies (kNN distance 0), which made the benign percentile scale
    optimistic: unseen benign traffic was flagged at 3.3% instead of 3.0%.
    """
    benign = train[train[config.LABEL_COLUMN] == config.BENIGN_LABEL]
    feats = [c for c in benign.columns if c not in config.METADATA_COLUMNS]
    benign = benign[~benign.duplicated(subset=feats, keep="first")]
    fit, ref = train_test_split(benign, test_size=config.BENIGN_REFERENCE_FRACTION,
                                random_state=seed, shuffle=True)
    return fit, ref


def eval_split(ev: pd.DataFrame, seed: int = config.RANDOM_SEED):
    cal, hold = train_test_split(ev, test_size=0.5, random_state=seed, shuffle=True,
                                 stratify=ev[config.LABEL_COLUMN])
    return cal, hold


def load_splits(seed: int = config.RANDOM_SEED) -> Splits:
    train = load_flows(config.TRAIN_PATH)
    fit, ref = benign_split(train, seed)
    del train
    ev = load_flows(config.EVAL_PATH)
    cal, hold = eval_split(ev, seed)
    return Splits(fit, ref, cal, hold)
