"""Configuration for Agent 2 (Anomaly Detection).

All tunable constants live here so that training, calibration, evaluation and
inference share a single source of truth.
"""
from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42

# --------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "raw"
TRAIN_PATH = DATA_DIR / "training" / "flows.parquet"
EVAL_PATH = DATA_DIR / "eval" / "flows.parquet"
MODEL_PATH = REPO_ROOT / "models" / "anomaly_detector.joblib"
OUTPUT_DIR = REPO_ROOT / "outputs"

# --------------------------------------------------------------------------- columns
ID_COLUMN = "record_id"
LABEL_COLUMN = "label"
METADATA_COLUMNS = (ID_COLUMN, LABEL_COLUMN)
BENIGN_LABEL = "BENIGN"
ATTACK_LABELS = ("BRUTE_FORCE", "DOS", "DDOS", "PORT_SCAN")

# Features excluded from the model by design (in addition to the constant /
# exact-duplicate columns detected automatically at fit time).
#   * Destination Port: a categorical identifier, not a magnitude. Treating it
#     as numeric makes "port 3690" look 40x more extreme than "port 80". The
#     task brief also states that endpoint fields are delivered separately in
#     the replay data, so Agent 2 must not rely on it. Ablation results are in
#     outputs/model_selection.json.
FORCED_EXCLUSIONS = {
    "Destination Port": "categorical endpoint identifier, not a numeric magnitude",
}

# A feature is dropped as near-constant when its most frequent value covers at
# least this share of the BENIGN fit data AND fewer than this many rows differ.
# (This removes e.g. 'Subflow Bwd Packets', which is non-zero in 3 of 456k rows.)
NEAR_CONSTANT_MAX_OTHER_ROWS = 10

# Robust-z values (standardised log space) are winsorised to this magnitude so a
# single pathological feature cannot dominate distance-based scores.
ROBUST_Z_CLIP = 50.0

# --------------------------------------------------------------------------- model
# Share of BENIGN training flows held out (never used for fitting) to build the
# reference distribution that maps raw scores to the normalised anomaly score.
BENIGN_REFERENCE_FRACTION = 0.2

# Model features are pruned (label-free) so that no two kept features have
# |Pearson r| >= this value in standardised benign log space. Redundant copies
# otherwise over-weight one behaviour in distances and IF split sampling.
MODEL_FEATURE_CORRELATION_PRUNE = 0.95
CORRELATION_SAMPLE_SIZE = 50_000

# Ensemble member 1: k-nearest-neighbour distance to benign prototypes
# (local density; handles the multi-modal benign population).
KNN_N_NEIGHBORS = 5
KNN_N_PROTOTYPES = 100_000

# Ensemble member 2: Isolation Forest (global outlyingness).
ISOLATION_FOREST_PARAMS = dict(
    n_estimators=300,
    max_samples=4096,
    max_features=1.0,
    bootstrap=False,
    n_jobs=-1,
    random_state=RANDOM_SEED,
)

# Calibration rule (see calibrate.py): choose the threshold that maximises F1
# on the eval calibration half subject to benign FPR <= this budget.
CALIBRATION_MAX_BENIGN_FPR = 0.03

# Default decision threshold on the normalised anomaly score. Overwritten by
# calibration (see calibrate.py); the calibrated value is stored in the model
# bundle and takes precedence at inference time.
DEFAULT_THRESHOLD = 0.99

# --------------------------------------------------------------------------- evidence
EVIDENCE_TOP_K = 5
# Severity is defined by how rare the observed value is under the BENIGN
# reference distribution (tail probability), not by hand-picked raw cut-offs.
SEVERITY_TAIL_PROBABILITY = (
    ("high", 0.001),    # outside the central 99.8% of benign flows
    ("medium", 0.01),   # outside the central 98%
    ("low", 0.05),      # outside the central 90%
)
