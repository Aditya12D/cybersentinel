# Agent 2 — Anomaly Detection (CyberSentinel)

Agent 2 independently judges how unusual a network flow is compared with **benign** traffic.
For each flow it returns an anomaly score, a flag, measured feature-level evidence and a reason.
It does **not** predict attack classes, does not use Agent 1's output, and never returns ALLOW/BLOCK.

## Quick start

```bash
pip install -r src/agents/anomaly_detection/requirements.txt

python -m src.agents.anomaly_detection.train            # fit + calibrate, ~20 s
python -m src.agents.anomaly_detection.evaluation       # ~1 min
python -m src.agents.anomaly_detection.ablation         # ~2 min
python -m src.agents.anomaly_detection.error_analysis   # ~30 s
python -m src.agents.anomaly_detection.model_selection  # ~3 min, optional
python -m pytest tests/anomaly_detection -q             # 36 tests, ~7 s
```

What each command produces:

| command | output |
|---|---|
| `train` | `models/anomaly_detector.joblib` (22 MB, git-ignored), `outputs/threshold_analysis.csv`, `outputs/calibration.json` |
| `evaluation` | `outputs/anomaly_evaluation.json`, `outputs/threshold_comparison_holdout.csv`, `outputs/score_distribution.png` |
| `ablation` | `outputs/ablation.json` (kNN-only vs Isolation-Forest-only vs fused) |
| `error_analysis` | `outputs/error_analysis.json` |
| `model_selection` | `outputs/model_selection.json` |

**Reproducibility:** all seeds are fixed (42). Two independent `train` runs produced a byte-identical model file and identical scores on all 114,185 eval rows. The model is not committed; re-create it with `train`.

## Agent 3 integration

```python
from src.agents.anomaly_detection import detect_anomaly, detect_anomalies

r  = detect_anomaly(flow_dict)        # one flow   -> dict
rs = detect_anomalies(df_or_dicts)    # many flows -> list[dict], same order as the input
```

- **Model loading:** the model loads once on first call, from `models/anomaly_detector.joblib`. Loading is thread-safe and cached. If the file is missing, run `train` first.
- **Input:** pass the raw flow, including `record_id`.
  - Endpoint fields, timestamps, `label` and Agent 1 output may be present; they are ignored.
  - Header whitespace in column names (e.g. `" Flow Duration"`) is tolerated.
- **Throughput:** batches cost about 0.19 ms per flow. A single call takes about 4.4 ms (p50) and 8.8 ms (p99).
- **Stricter threshold:** `AnomalyDetectionAgent(threshold=0.99)` gives the strict operating point.
- **Using the result:** treat `is_anomaly` and `anomaly_score` as one independent signal. Weight results with `data_quality.reliable == false` down.
- **Errors:**
  - `TypeError` if the input is not a mapping, Series or DataFrame.
  - `ValueError` if a flow contains no usable model feature.

## Output schema (schema_version 1.1)

Every flow returns the same keys, and the result is JSON-serialisable.

| key | meaning |
|---|---|
| `record_id` | Echoed from the input as a string, or `null`. |
| `anomaly_score` | A value in [0, 1]: the **benign percentile**, i.e. the fraction of held-out, de-duplicated benign reference flows that are no more unusual than this flow. A score of 0.99 means "more unusual than 99% of normal traffic". **It is not a probability of attack.** |
| `is_anomaly` | `anomaly_score >= threshold`. |
| `evidence[]` | `feature`, `observed_value`, `reference` (text), `severity` (high / medium / low / info), `direction`, `benign_tail_probability`, `benign_median`, `benign_p01`, `benign_p99`, `nearest_benign_range`, `distance_contribution` |
| `reason` | Deterministic text built only from the score, the components and the evidence. |
| `rationale` | Alias of `reason`, kept for backward compatibility. |
| `threshold` | The calibrated threshold, 0.97. |
| `components` | `knn_benign_percentile` and `isolation_forest_benign_percentile`. |
| `data_quality` | See below. |
| `agent`, `schema_version`, `model_version` | `"anomaly_detection"`, `"1.1"`, `"agent2-knn-iforest-1.1"`. |

`data_quality` fields:

| field | meaning |
|---|---|
| `missing_features` | Feature absent from the input, or its value is null or NaN. |
| `invalid_features` | Value present but not numeric. |
| `non_finite_features` | Value is +inf or −inf. |
| `imputed_model_features` | Model features whose value was repaired rather than observed. |
| `reliable` | `true` when at most 10% of the model features were imputed. |

Imputed values are **never** reported as evidence. When repairs happen, the reason text says so.

## Architecture

```
raw flow -> FlowPreprocessor (fit on benign only)
         -> 69 features -> label-free correlation pruning (|r| < 0.95) -> 40 model features
         -> kNN mean distance to 5 nearest of 100k benign prototypes  -> benign percentile p_knn
         -> Isolation Forest (300 trees, max_samples 4096)          -> benign percentile p_if
anomaly_score = benign percentile of (p_knn + p_if) / 2
```

### Data and splits

- **Training data:** only BENIGN training rows are used. Exact feature-duplicate rows are removed first, leaving 229,647 rows.
- **Benign split:**
  - 80% "fit" (183,717 rows) for the preprocessor, the prototypes and the Isolation Forest.
  - 20% "reference" (45,930 rows), never fitted, used for score percentiles and the evidence distributions.
  - Deduplicating before the split guarantees that no reference flow is a memorised copy of a fit flow.
- **Eval labels** are used only for threshold calibration and evaluation.
  - Eval is split 50/50, stratified by label, with seed 42.
  - The **calibration** half is used for every choice.
  - The **holdout** half is used only for reporting.

### Preprocessing (`preprocessing.py`)

1. **Columns:**
   - Drop the metadata: `record_id` and `label`.
   - Drop `Destination Port`. It is categorical, endpoint fields arrive separately in the replay data, and the ablation showed no recall benefit from keeping it.
   - Drop constant columns, near-constant columns and exact-duplicate columns.
2. **Repairs:**
   - NaN or missing values are replaced by the benign median.
   - +inf and −inf are replaced by the maximum and minimum of the fit data.
   - Non-numeric values are replaced by the benign median and reported as invalid.
3. **Transform:** signed `log1p`, then a robust z-score (median and IQR/1.349, falling back to the standard deviation), winsorised to ±50.

### Inference optimisations (exact, verified by tests)

- **Isolation Forest:** for 512 rows or fewer, a vectorised traversal of all trees (`fast_iforest.py`) is used. Its scores are bit-identical to sklearn's, and it is 10–20× faster for a single flow.
- **kNN:**
  - Neighbours are found with a kd-tree for 8 rows or fewer and with brute force otherwise.
  - Distances to the found neighbours are recomputed in float64, so the single-flow and batch paths agree bit-for-bit.
  - This also avoids sklearn's ‖a‖² − 2ab + ‖b‖² cancellation error.
- **Model bundle:** derived caches (the kd-tree, the compiled forest and the evidence reference) are rebuilt on load rather than pickled.

## Threshold calibration (`calibrate.py`)

**Rule:** maximise F1 on the calibration half subject to **benign FPR ≤ 3%**, breaking ties towards the higher threshold. Accuracy is never optimised.

**Selected threshold: 0.97.** On the calibration half it gives FPR 2.77%, recall 0.9927 and F1 0.9812.

**Stored operating points:**

| point | rule | threshold |
|---|---|---|
| strict | FPR ≤ 1% | 0.99 |
| balanced | FPR ≤ 3% (default) | 0.97 |
| sensitive | FPR ≤ 5% | 0.97 |

`sensitive` equals `balanced` because the F1 maximum already falls inside the 3% budget.

## Results (holdout half, untouched; threshold 0.97)

**Overall (n = 57,093):**

| precision | recall | F1 | benign FPR | accuracy | ROC-AUC | PR-AUC (AP) |
|---|---|---|---|---|---|---|
| 0.9680 | 0.9927 | 0.9802 | 0.0297 | 0.9810 | 0.9894 | 0.9692 |

**Confusion matrix:**

| | predicted normal | predicted anomalous |
|---|---|---|
| BENIGN | TN = 29,111 | FP = 890 |
| attack | FN = 197 | TP = 26,895 |

**Per-class detection:**

| class | detected |
|---|---|
| BRUTE_FORCE | 689 / 692 (99.6%) |
| DOS | 11,806 / 12,000 (98.4%) |
| DDOS | 7,200 / 7,200 (100%) |
| PORT_SCAN | 7,200 / 7,200 (100%) |

**Other views:**
- **Holdout without rows that duplicate a training row:** F1 0.9779, FPR 0.0318.
- **Full eval partition** (half of it informed the threshold): F1 0.9807, FPR 0.0287.

**Threshold comparison (holdout):**

| threshold | precision | recall | F1 | FPR | BF | DoS | DDoS | PS |
|---|---|---|---|---|---|---|---|---|
| 0.90 | 0.902 | 1.000 | 0.949 | 0.098 | 0.999 | 1.000 | 1.000 | 1.000 |
| 0.95 | 0.948 | 0.999 | 0.973 | 0.050 | 0.999 | 0.998 | 1.000 | 1.000 |
| **0.97** | 0.968 | 0.993 | 0.980 | 0.030 | 0.996 | 0.984 | 1.000 | 1.000 |
| 0.98 | 0.977 | 0.908 | 0.941 | 0.019 | 0.652 | 0.814 | 0.997 | 1.000 |
| 0.99 | 0.982 | 0.576 | 0.726 | 0.010 | 0.568 | 0.280 | 0.659 | 0.986 |

### Ablation (`ablation.py`)

All variants use the same splits, the same reference set and the same calibration rule. Results are on the holdout.

| variant | recall | FPR | F1 | ROC-AUC | PR-AUC | PORT_SCAN | DOS | BRUTE_FORCE |
|---|---|---|---|---|---|---|---|---|
| kNN only | 0.787 | 2.89% | 0.866 | 0.987 | 0.980 | 0.202 | 0.999 | 0.999 |
| Isolation Forest only | 0.286 | 2.82% | 0.434 | 0.945 | 0.873 | 1.000 | 0.045 | 0.000 |
| **fused (deployed)** | **0.993** | **2.97%** | **0.980** | 0.989 | 0.969 | 1.000 | 0.984 | 0.996 |
| fused, all 184k prototypes | 0.995 | 2.91% | 0.982 | 0.991 | 0.975 | 1.000 | 0.989 | 0.996 |

- The two members are complementary. kNN carries DoS, DDoS and brute force; Isolation Forest carries PORT_SCAN.
- Using all prototypes changes FPR by only 0.06 points, which is within noise, and doubles kNN cost. It was rejected.

### Latency (this machine)

| path | Isolation Forest | kNN | end-to-end |
|---|---|---|---|
| single flow | 0.8 ms | 1.5 ms | 4.4 ms p50, 8.8 ms p99 |
| batch | 28 ms per 1k rows | 160 ms per 1k rows | 0.19 ms per flow |

Model load takes about 1 s.

## Evidence (`evidence.py`)

Two measured quantities are computed per model feature. No explanations are hand-written per attack type.

1. **`distance_contribution`** is the exact share of the squared kNN distance to the 5 nearest benign prototypes that is due to this feature. It is grounded in the model.
2. **`benign_tail_probability`** is `min(P_benign(X ≥ x), P_benign(X ≤ x))` on the benign reference set. It is grounded in the benign distribution.

**Severity** is derived only from the tail probability:

| severity | tail probability |
|---|---|
| high | ≤ 0.001 |
| medium | ≤ 0.01 |
| low | ≤ 0.05 |
| info | > 0.05 |

`info` means the value is common among benign flows, but this feature still separates the flow from its nearest benign neighbours. In other words, the anomaly lies in the combination of values.

**Selection:**
- **Anomalous flows:** the top contributors until 80% of the distance is explained (at most 5 features, each with at least a 5% share). Remaining slots are filled with the rarest values of severity medium or higher.
- **Normal flows:** only individually rare values (medium or high) are listed.

**Tested properties:**
- `observed_value` always equals the input value.
- Imputed features never appear as evidence.
- Severity is always consistent with the tail probability.

## Error analysis (`error_analysis.py`, holdout)

### False positives

There are 890 false positives in total.

**By destination port** (used for analysis only, never as a model input):

| port | benign FPR | share of all FPs | share of benign traffic |
|---|---|---|---|
| 443 (HTTPS) | 8.5% | 67% | 23% |
| 80 (HTTP) | 6.5% | 24% | 11% |
| 53 (DNS) | 0.08% | — | 62% |

**HTTPS false positives are long-lived sessions:**

| | median duration | median packets (fwd/bwd) | median bwd bytes |
|---|---|---|---|
| falsely flagged HTTPS | 62 s | 33 / 39 | 16 KB |
| correctly passed HTTPS | 5.8 s | 12 / 11 | 4.4 KB |

- Long web sessions are sparse in the benign training data.
- kNN alone raises 289 of the 593 HTTPS flags.

**Port 443 is not suppressed.** The holdout contains only 8 attack flows on 443, so suppression would look free on this data. But nothing shows that 443 traffic is safe, and suppression would hide HTTPS-borne attacks in the replay.

### Missed attacks

- **DOS: 194 missed.** Their median score is 0.966, just below the threshold. kNN ranks them high (0.985), but Isolation Forest is low (0.865) and pulls the fused score down.
  - 43% last longer than 5 s, against 9% of detected DOS.
  - 39% have fewer than 2 packets/s, consistent with slow-rate DoS.
  - Slow, long flows also make up 21% of benign traffic, with a 6.5% FPR inside that group. That overlap is why these misses hide.
- **BRUTE_FORCE: 3 missed.** All have no RST flags, and they look like ordinary interactive sessions.

## Files

```
src/agents/anomaly_detection/
  config.py          constants, seeds, paths
  preprocessing.py   FlowPreprocessor, input validation, ImputationReport
  datasets.py        deterministic, deduplicated benign fit/reference + eval calibration/holdout splits
  detector.py        AnomalyDetector (kNN + Isolation Forest, normalisation, save/load)
  fast_iforest.py    exact vectorised Isolation Forest scorer
  evidence.py        evidence + reason text
  calibrate.py       threshold sweep and selection
  metrics.py         binary and per-class metrics
  evaluation.py      holdout / full-eval report
  ablation.py        kNN-only vs IF-only vs fused
  error_analysis.py  false-positive / missed-attack analysis
  model_selection.py candidate comparison (IF, OC-SVM, statistical, kNN, ensembles)
  train.py           fit + calibrate + save
  api.py             detect_anomaly / detect_anomalies / AnomalyDetectionAgent
tests/anomaly_detection/   36 tests (synthetic unit tests + 1 integration test on the real model)
```

## Known limitations

- **Small score margin.** Attack scores sit between 0.96 and 0.995, so a shift in the replay's benign mix moves FPR and DoS / brute-force recall. More long HTTPS sessions would raise FPR above 3%.
- **Precision depends on prevalence.** The eval set is 47% attacks. At a 5% attack prevalence, the same FPR and recall would give a precision of about 0.64.
- **Eval data was used for selection.** Threshold and model choices used the eval calibration half. The holdout numbers are unbiased but come from the same distribution.
- **No cross-flow context.** Each flow is scored independently, so behaviour such as one source scanning many ports is invisible.
