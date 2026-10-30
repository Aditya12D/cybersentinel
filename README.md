# CyberSentinel — JAI 2026 Track 1

Multi-agent network intrusion detection on CICIDS2017 (5 classes).

## Setup
```bash
uv venv -p 3.13 .venv && uv pip install -p .venv/bin/python -r requirements.txt
# put the participant pack at data/Cyber/{training,eval}/flows.parquet (gitignored)
```

## Agent 1 — Threat Detection (`agents/threat_detection.py`)

```python
from agents.threat_detection import predict_threat, predict_batch, predict_proba_frame
predict_threat(flow_dict)   # one flow (dict of the 78 feature columns; record_id/label ignored)
predict_batch(dataframe)    # list of the same dicts, one per row
predict_proba_frame(df)     # raw (n, 5) probability array, column order = CLASSES
```
Output:
```json
{"agent": "threat_detection", "predicted_class": "BRUTE_FORCE", "confidence": 0.999909,
 "probabilities": {"BENIGN": 8e-05, "BRUTE_FORCE": 0.999909, "DOS": 2e-06, "DDOS": 4e-06, "PORT_SCAN": 5e-06}}
```
Class order is fixed: `BENIGN, BRUTE_FORCE, DOS, DDOS, PORT_SCAN`. `confidence` = max model probability.

**Preprocessing** (same at train and inference): fixed column order, float32, ±inf → NaN, unparseable → NaN,
missing columns → NaN (LightGBM routes NaN natively). Dropped 7 zero-information columns: 3 constant
(`Fwd/Bwd URG Flags`, `URG Flag Count`) and 4 exact copies (`Fwd Header Length_duplicated_0`,
`Avg Fwd/Bwd Segment Size`, `Average Packet Size`). `record_id` never used.

**Data findings:** no NaN/inf in the pack (handled anyway for the hidden set); 24,132 duplicate feature
rows in train, none with conflicting labels; **9,795 eval rows exactly match a train row**, so we also
report "unseen" eval rows only.

**Model selection** — stratified 15% validation split of training data; eval never fitted on:

| Model | Val macro-F1 | Eval acc | Eval macro-F1 | Unseen macro-F1 | BRUTE_FORCE recall | Train s |
|---|---|---|---|---|---|---|
| Extra Trees | 0.99985 | 0.99996 | 0.99990 | 0.99989 | 0.99928 | 21 |
| Random Forest | 0.99985 | 0.99996 | 0.99990 | 0.99989 | 0.99928 | 27 |
| **LightGBM** | 0.99987 | 0.99998 | **0.99998** | 0.99997 | **1.00000** | 71 |
| XGBoost | 0.99987 | 0.99998 | 0.99998 | 0.99997 | 1.00000 | 39 |
| CatBoost | 0.99999 | 0.99997 | 0.99991 | 0.99990 | 0.99928 | 947 |

Differences are 1–5 rows out of 114,185, i.e. within noise. LightGBM chosen: tied-best eval, perfect
BRUTE_FORCE recall unweighted (so no class weighting needed), lowest log-loss (best-behaved probabilities),
fast retraining. Final model refit on all training rows (`python train_agent1.py final --model lgbm`).

**Final eval:** accuracy 0.99998, macro-F1 0.99998, ECE 0.00001. Confusion: only 2 errors,
both PORT_SCAN → BENIGN. Artifacts in `models/` and `evaluation/`.

**Note for Agent 3:** every error seen across all models was an attack predicted as BENIGN, usually with
confidence < 0.9 — a low-confidence BENIGN is the case where Agent 2's anomaly score should matter most.

**Generalization check** (`python train_agent1.py cv`): stratified 5-fold CV on the 432,605 *unique*
training rows (duplicates removed so identical flows can't straddle folds; fixed tree counts, no early
stopping on the test fold). Results in `evaluation/cv_results.csv`.

| Model | Errors / fold (~86.5k rows) | Macro-F1 mean ± std | BRUTE_FORCE recall |
|---|---|---|---|
| **LightGBM** | 2.4 ± 1.1 | **0.99994 ± 0.00006** | 0.99982 |
| XGBoost | 2.6 ± 1.5 | 0.99992 ± 0.00006 | 0.99982 |
| LightGBM+XGBoost avg | 2.8 ± 0.8 | 0.99992 ± 0.00005 | 0.99982 |

The ensemble does not beat LightGBM consistently → rejected. Production model unchanged.

**Error analysis** (`python analyze_agent1_errors.py`, read-only → `evaluation/error_analysis.json`):
both eval errors (C023985, C024161) are multi-second flows with no SYN/RST and zero init window,
nothing like a ~48 µs SYN→RST scan probe; 23 of their 25 nearest training rows are BENIGN. Likely
CICIDS2017 time-window labelling of background flows from the scanner host, so no generalizable
fix exists at single-flow level and none was applied. Only 4 eval rows have confidence < 0.99 and both
errors are among them (0 errors above 0.99) → Agent 3 can treat confidence < 0.99 as "uncertain".
Port scans are a multi-flow pattern (one source, many ports): catch them with the hidden set's
IP/port metadata in fusion, not by memorizing rows.

Tests: `python -m pytest tests -q` (contract, dirty input, batch == single). Single-flow latency ≈ 2 ms.
