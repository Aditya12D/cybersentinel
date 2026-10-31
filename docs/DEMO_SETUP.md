# Running CyberSentinel (and the demo) on another PC

Tested on macOS 26 (Apple Silicon), Python 3.13.9. Everything is plain Python; no GPU, no environment
variables, no internet needed after installation.

## 1. Get the code
```bash
git clone https://github.com/Aditya12D/cybersentinel.git
cd cybersentinel
git checkout feature/level2-agents4-5      # until merged into main
```

## 2. Python environment (Python 3.13; 3.11+ should work but is untested)
```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
# Windows PowerShell (if activation is blocked: Set-ExecutionPolicy -Scope Process Bypass)
py -3.13 -m venv .venv ; .venv\Scripts\Activate.ps1

pip install -r requirements.txt            # exact pinned versions; the saved Agent 1 model needs them
```
Use a phone hotspot if the venue Wi-Fi breaks `pip` (it intercepts TLS).

## 3. Data (never committed — `data/` is gitignored)
```
data/raw/training/flows.parquet         <- participant pack (Agent 2 training needs it)
data/raw/eval/flows.parquet             <- participant pack (tests + demo section 3)
data/hidden/Cyber/hidden/records.csv    <- unzip cyber-hidden.zip into data/hidden/
```
```bash
unzip cyber-hidden.zip -d data/hidden       # Windows: Expand-Archive cyber-hidden.zip data\hidden
```

## 4. Models
| Model | In git? | How to get it |
|---|---|---|
| Agent 1 `models/threat_model.pkl` (LightGBM) | **yes** | nothing to do |
| Agent 2 `models/anomaly_detector.joblib` | **no** (22 MB) | **Best for the demo:** copy the file from the Mac (USB / Drive) — sha256 starts `2510388d930b94e5`. Alternative: `python -m src.agents.anomaly_detection.train` (~10 s; needs both parquet files). |

A retrain on a different OS or library build can differ in the last digits of the anomaly scores, so
`classification.csv` would no longer be byte-identical to the frozen submission (decisions should not change).
Copying the Mac model avoids this.

## 5. Verify
```bash
python -m pytest tests -q          # expect: 153 passed
python level2.py                   # writes outputs/level2/*.csv and validates them (valid=True)
python validate_submission.py      # VALID
python demo.py 6                   # last table must say IDENTICAL for all three files
```

## 6. Run the demo
```bash
python demo.py          # menu: 1-5 one agent each, 6 full pipeline, 7 policy lifecycle, 8 dashboard
python demo.py all      # everything, ~6 s
```
Expected key lines: section 7 shows `[TRIGGER] F000014 ... created P0001`, `[FIRST BLOCK] F000018 ... BLOCK P0001`,
`[AFTER EXPIRY] F000115 ... created P0002`; section 8 shows 200 records, 98 BLOCK, 2 policies.

## Common problems
| Symptom | Fix |
|---|---|
| `FileNotFoundError: Model not found at models/anomaly_detector.joblib` | Step 4 (copy or train Agent 2) |
| `Hidden records not found` | Step 3 (unzip into `data/hidden/`) |
| `ModuleNotFoundError: src` / `agents` | Run commands from the repo root |
| `pip` SSL / certificate errors | Venue Wi-Fi TLS interception → use a hotspot |
| LightGBM import error on macOS (`libomp`) | `brew install libomp` |
| Demo section 6 says DIFFERENT | Agent 2 was retrained on another machine → copy the Mac model (step 4) |
| Different SHA-256 for files written by `level2.py` on **Windows** | pandas writes Windows line endings (CRLF) there; the content is the same. Submit the frozen files in `submission/level2_v1/` (the demo compares content with Unix line endings). |
| Unpinned versions from `src/agents/anomaly_detection/requirements.txt` | Install only the root `requirements.txt` |

**Submit only `submission/level2_v1/*.csv`** (checksums in `SHA256SUMS`); never a fresh run from another
machine.
