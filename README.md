# CyberSentinel — JAI 2026 Track 1

Multi-agent network intrusion detection on CICIDS2017 (5 classes).

## Setup
```bash
uv venv -p 3.13 .venv && uv pip install -p .venv/bin/python -r requirements.txt
# put the participant pack at data/raw/{training,eval}/flows.parquet (gitignored; both agents read it)
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

## Agent 2 — Anomaly Detection (`src/agents/anomaly_detection/`)

Independent, benign-only novelty detector. It reports how unusual a flow is compared with normal
traffic, with measured feature evidence and a reason. It never predicts an attack class, never uses
Agent 1's output and never returns ALLOW/BLOCK. Full documentation:
[`src/agents/anomaly_detection/README.md`](src/agents/anomaly_detection/README.md).

```bash
pip install -r src/agents/anomaly_detection/requirements.txt
# expects data/raw/{training,eval}/flows.parquet (gitignored)
python -m src.agents.anomaly_detection.train        # ~20 s -> models/anomaly_detector.joblib (gitignored, deterministic)
python -m src.agents.anomaly_detection.evaluation   # -> outputs/anomaly_evaluation.json
python -m pytest tests/anomaly_detection -q         # 36 tests
```

```python
from src.agents.anomaly_detection import detect_anomaly, detect_anomalies
detect_anomaly(flow_dict)      # one flow -> dict
detect_anomalies(df_or_dicts)  # list of dicts, input order preserved (preferred for throughput)
```
Output (abridged):
```json
{"record_id": "C289947", "anomaly_score": 0.992771, "is_anomaly": true, "threshold": 0.97,
 "evidence": [{"feature": "RST Flag Count", "observed_value": 2.0, "severity": "low", "direction": "above",
               "reference": "Above benign reference range: only 3.87% of benign flows have a value this high or higher ..."}],
 "reason": "Anomaly score 0.993: more unusual than 99.3% of benign reference flows ...",
 "components": {"knn_benign_percentile": 0.9989, "isolation_forest_benign_percentile": 0.9493},
 "data_quality": {"missing_features": [], "invalid_features": [], "non_finite_features": [],
                  "imputed_model_features": [], "reliable": true},
 "agent": "anomaly_detection", "schema_version": "1.1"}
```
`anomaly_score` is a **benign percentile** (0.99 = more unusual than 99% of normal traffic), not an attack probability.

**Model:** the average of two benign percentiles, fitted on de-duplicated BENIGN training flows only.
- **kNN distance** to the 5 nearest of 100k benign prototypes. It catches DoS, DDoS and brute force.
- **Isolation Forest.** It catches port scans.

Both members use 40 de-correlated features; `Destination Port` is excluded. The threshold of 0.97
maximises F1 subject to a benign false-positive rate of at most 3%, chosen on half of eval; the other
half is used only for reporting.

**Holdout results** (untouched half of eval, n = 57,093):

| precision | recall | F1 | benign FPR | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 0.968 | 0.993 | 0.980 | 2.97% | 0.989 | 0.969 |

Per-class recall: BRUTE_FORCE 99.6%, DOS 98.4%, DDOS 100%, PORT_SCAN 100%.

**Ablation** (same splits and rule): kNN alone reaches recall 0.787 (20% of port scans), Isolation
Forest alone 0.286, and the fused model 0.993.

**Latency:** single flow 4.4 ms (p50), batch 0.19 ms per flow.

**Note for Agent 3:**
- False positives concentrate in long-lived benign HTTPS/HTTP sessions (port 443 FPR 8.5%; port 53 FPR 0.08%).
- Misses are slow, long DoS flows scoring just under the threshold (median score 0.966).
- Agent 2 is most useful exactly where Agent 1 is a low-confidence BENIGN. Treat `data_quality.reliable == false` with caution.

## Agent 3 — Evidence Fusion (`src/agents/evidence_fusion.py`) + pipeline (`run_pipeline.py`)

Deterministic, rule-based fusion of Agent 1 and Agent 2 (no LLM, no model, no randomness). Both
agents' outputs are validated strictly first; Agent 2 can never supply or change an attack class,
an Agent 2 "normal" never downgrades an Agent 1 attack, and invalid evidence never yields ALLOW.

```bash
python -m src.agents.anomaly_detection.train   # once: Agent 2 model is not in git (~10 s)
python run_pipeline.py                         # eval set -> outputs/pipeline_report.json (+ per-flow csv/jsonl, gitignored)
python run_pipeline.py --input hidden/flows.parquet   # any flows file; labels optional
python -m pytest tests -q                      # Agents 1 + 2 + 3, unit + end-to-end
```
```python
from src.agents.evidence_fusion import fuse, fuse_batch
fuse(predict_threat(flow), detect_anomaly(flow), record_id=flow["record_id"])
```

| rule | condition | final_class | decision | evidence_status |
|---|---|---|---|---|
| invalid_threat_evidence | Agent 1 missing / raised / malformed | UNKNOWN | BLOCK | INVALID_EVIDENCE |
| invalid_anomaly_evidence | Agent 2 missing / raised / malformed | Agent 1 class | BLOCK | INVALID_EVIDENCE |
| unreliable_anomaly | Agent 2 `data_quality.reliable == false` | Agent 1 class | BLOCK | UNCERTAIN |
| low_confidence | Agent 1 confidence < 0.99 | Agent 1 class | BLOCK | UNCERTAIN |
| attack_agreement | attack + anomalous | Agent 1 class | BLOCK | AGREEMENT |
| attack_over_normal | attack + normal | Agent 1 class | BLOCK | DISAGREEMENT |
| benign_over_anomaly | confident BENIGN + anomalous | BENIGN | ALLOW (`needs_review`) | DISAGREEMENT |
| benign_agreement | confident BENIGN + normal | BENIGN | ALLOW | AGREEMENT |

`needs_review` is true for every status except AGREEMENT (the internal "REVIEW" flag; the policy output
stays ALLOW/BLOCK). All thresholds and their training-only justification are in `FusionConfig`.
An experimental Bayes escalation rule exists but is **off by default** (fired on 1 of 248,888
training-only out-of-fold rows — not enough evidence to enable).

**Eval set (n = 114,185; no fusion rule was tuned on it):** final macro-F1 0.99998 (same 2 PORT_SCAN
rows as Agent 1, both now **BLOCKed** as UNCERTAIN), 0 false ALLOW, 1 false BLOCK, 2,088 flows flagged
for review. Throughput 0.19 ms/flow in batch; 5.7 ms p50 for a single flow through all three agents.
