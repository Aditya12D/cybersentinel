# Pre-merge Final Audit — `feature/level2-agents4-5`

31 Oct 2026, 12:05. Branch head `11137df` (implementation frozen at `bcc9697`; later commits add only demo
files). Nothing committed, pushed or merged by this audit; `submission/level2_v1/` untouched (checksums OK).
No model, threshold, fusion rule, correlation logic or policy duration changed. Evidence:
`outputs/pre_merge_candidates/premerge_report.json`.

## Task 1 — Official schema

Sources, newest and most authoritative first:
1. **Portal stage page** (printed 31 Oct 10:21 from `c-thon.vercel.app/cyber/L2`, section *"RULES THE VALIDATOR
   ENFORCES"* — the scored interface).
2. Hidden-data `README.md` shipped inside `cyber-hidden.zip`.
3. `Track 1 Problem Statement (2).docx` — pre-event document (still says hidden data is released at 1:30 PM,
   whereas the portal released it at 10:00 AM).

| Question | Portal (scored interface) | README | Problem statement (docx) | Conclusion |
|---|---|---|---|---|
| Is `SRC_DST_PAIR` accepted? | *"scope_type is one of SOURCE_IP, DESTINATION_IP, SRC_DST_PAIR … pairs use source\|destination"* | same three scopes | *"For the scored interface, scope_type = SOURCE_IP"* | **Accepted** per the portal validator rules |
| Is `SOURCE_IP` required? | no — one of three | no | yes | **Not required** by the portal; docx conflicts |
| Is `DESTINATION_IP` accepted? | yes | yes | no | **Accepted** (we never use it) |
| `active_policy_id` for ALLOW | *"ALLOW uses NONE"*; *"An ALLOW must report NONE and must not have a matching active policy"*; pseudo-code *"active_policy_id = matches ? min(matches) : NONE"* | — | *"ALLOW has an empty applied_policy_id"* | **`NONE`** per the portal |
| `policies.csv` columns | template: 5 columns, no `policy_action` | — | 6 columns incl. `policy_action` | **5 columns** (portal template, header byte-identical) |

The docx is also internally inconsistent (it lists policy columns under `replay.csv` and names a column
`applied_policy_id` that exists in no template). Our files follow the portal on every point.

**Residual risk (UNRESOLVED in the strict sense):** the portal's validator *code* is not inspectable; its stated
rules are the best available evidence. If an upload is rejected on scope, a SOURCE_IP variant is ready (Task 3) and
is decision-identical.

## Task 2 — CSV portability

| Property (frozen files) | classification | replay | policies |
|---|---|---|---|
| Line endings | LF (0 CRLF) | LF | LF |
| UTF-8, BOM | UTF-8, no BOM | UTF-8, no BOM | UTF-8, no BOM |
| Quoting / trailing spaces | none / none | none / none | none / none |
| Header = portal template | yes | yes | yes |
| Data rows | 200 | 200 | 2 |
| Record IDs exact set | yes | yes | — |

**Root cause of the Windows difference:** `level2.py` calls `DataFrame.to_csv(path, index=False)`; pandas uses
`os.linesep`, i.e. CRLF on Windows. Second source: Git for Windows defaults to `core.autocrlf=true`, which converts
the committed LF files in `submission/level2_v1/` to CRLF on checkout — so their SHA-256 no longer matches
`SHA256SUMS` on a Windows clone.

Simulation (pandas writer with `os.linesep = "\r\n"`), outputs in `outputs/pre_merge_candidates/`:

| | current code on Windows | with proposed LF fix |
|---|---|---|
| CRLF lines | 201 / 201 / 3 | 0 |
| Bytes identical to frozen | no | **yes (all 3)** |
| Parsed content identical to frozen | yes | yes |
| First record id parsed | `F000001` (no stray `\r`) | `F000001` |
| Offline validator | VALID | VALID |

**Proposed fix (not applied):** `outputs/pre_merge_candidates/lf_fix.patch` — add `lineterminator="\n"` to the
three `to_csv` calls in `level2.py`. **Proposed `.gitattributes`** (`proposed.gitattributes`): `*.csv -text` and
`SHA256SUMS -text`, so Git never rewrites line endings of submission files or checksums on any OS. `-text` (rather
than `eol=lf`) keeps the bytes exactly as committed, which is what the checksums certify.

## Task 3 — Policy scope sensitivity (hidden records only; not a hidden-label performance measure)

| Scope used by Agent 5 | BLOCK | ALLOW | Records changed vs submitted | Predicted-BENIGN records newly blocked | Files vs frozen |
|---|---|---|---|---|---|
| SRC_DST_PAIR (submitted) | 98 | 102 | — | — | identical |
| SOURCE_IP | 98 | 102 | **none** | none | classification + replay identical; policies differ only in scope_type/value; VALID |
| DESTINATION_IP (information only) | 99 | 101 | F000054 ALLOW→BLOCK | **F000054** | — |

All 100 flows from 172.16.0.1 go to 192.168.10.50, so pair and source scope are decision-identical here.
DESTINATION_IP would block an unrelated client of that server — the reason Agent 5 never uses it.
SOURCE_IP candidate files: `outputs/pre_merge_candidates/source_ip_scope/`.

## Task 4 — Policy lifecycle / expiry

| Record | Action | Why |
|---|---|---|
| F000014 | ALLOW, creates P0001 | First flow Agent 3 marks for BLOCK in the whole replay (A1 BRUTE_FORCE 0.999999; A2 0.980, flagged anomalous). No policy existed; a policy can only act on later records. |
| F000018 | BLOCK P0001 | First later flow matching `172.16.0.1\|192.168.10.50`; checked before analysis, so Agent 4 is skipped. |
| F000114 | BLOCK P0001 | P0001's expiry record — expiry is inclusive. |
| F000115 | ALLOW, creates P0002 | P0001 ended at F000114; nothing active matches, so the flow is analysed (A1 DOS 1.0; A2 0.988 flagged) and P0002 is created. |
| F000116 | BLOCK P0002 | First record after P0002's trigger. |

Agent 4 does not see blocked flows because the replay loop (`level2.replay_loop`) only calls `corr.observe` when
`defense.check` returns ALLOW — the brief's rule *"if ALLOW, analyse it"*. **Policy renewal is not supported**:
`AdaptiveDefense` creates policies only in `consider()`, which runs only for allowed flows, and has no extend or
renew path. This *policy-expiry gap* costs one predicted-attack flow per expiry (F000115). It is documented, not
tuned — changing the 100-record duration after seeing the replay would be tuning on hidden data. Future option:
renewal from blocked-traffic monitoring, validated on labelled data first.

## Task 5 — Regression (all executed)

| Check | Result |
|---|---|
| Full test suite | **PASS — 153 passed** (the 152 audited + 1 demo smoke test) |
| Agent 1–3 tests / Agent 4–5 tests | PASS (included above) |
| Full 200-record replay + offline validator | PASS — valid=True, VALID |
| Prefix replay (no look-ahead) | PASS (`test_prefix_replay_never_depends_on_future_records`) |
| Shuffled input → identical output | PASS (`test_hidden_replay_is_valid_and_deterministic`) |
| IP renaming → identical decisions | PASS |
| Single-flow vs batch (all 200 hidden records) | PASS — identical Agent 3 outputs |
| CSV schema vs portal templates | PASS (headers byte-identical; 200/200/2 rows) |
| Policy lifecycle F000014/18/114/115/116 | PASS (table above) |
| SHA-256 frozen files vs `SHA256SUMS` | PASS — 3/3 OK |
| Live LF output vs frozen | PASS — byte-identical |
| Windows CRLF output vs frozen | bytes differ (expected), content identical; fixed by the proposed patch |

Checksums:

| File | Frozen / live LF | Windows CRLF (current code) |
|---|---|---|
| classification.csv | `82dc963d411f6938…fcab6` | `fc63d0d7a9fff11b…e2958` |
| replay.csv | `300bf70333540dd5…30077` | `6745339b0c2e2573…7bb4e` |
| policies.csv | `4a1d2432299380c5…bd343`* | `77429d2f35419f29…ae734` |

*full value: `4a1d2432299380c5a0862066c2d86efb066dbf53d26e8c999461720e2a5bd343`.

## Unresolved official-interface questions
1. Validator source code is not available; conclusions rest on the portal's published rules (strong evidence).
2. Whether the scorer treats SRC_DST_PAIR and SOURCE_IP differently beyond validation (portal says only that the
   validator recomputes replay from the ledger; decisions here are identical either way).
3. Portal status was "Submissions are paused" at 10:21; a real upload is the only definitive acceptance test.

## Also pending (documentation, safe)
- README test counts are stale (says 133 / 150; now 153).
- `outputs/pipeline_report.json` example rationales predate the terminology fix (numbers unchanged); regenerate with
  `python run_pipeline.py` when convenient.

## Recommendation
**DO NOT MERGE yet — SAFE TO MERGE after two small approved changes:** apply `lf_fix.patch` and add the
`.gitattributes` (both verified: byte-identical output, VALID, no logic change), re-run the 153 tests, then
fast-forward `main`. Meanwhile **upload only the frozen Mac files in `submission/level2_v1/`** (LF, verified),
never a regenerated Windows copy. Keep SRC_DST_PAIR (portal-compliant); use the SOURCE_IP candidate only if the
portal rejects the scope.

## Decision (31 Oct, 12:10)
The team chose to merge `feature/level2-agents4-5` into `main` as verified (153 tests, frozen CSVs VALID and checksums OK). The LF line-ending patch and `.gitattributes` remain **pending**; until then, submit only the frozen Mac files in `submission/level2_v1/`, and on Windows compare CSV *content*, not raw checksums.
