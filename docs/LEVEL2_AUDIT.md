# Level 2 — Final Agent 4 / Agent 5 Audit (31 Oct, 11:00)

Audited commit: `d8312ba` + wording-only fixes. No model, threshold or policy logic was changed. Submission
candidate frozen in `submission/level2_v1/` (SHA-256 in `SHA256SUMS`); `python level2.py` reproduces it
byte-for-byte.

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Agent 4 never makes blocking decisions | PASS | `temporal_correlation.py` outputs groups, evidence strength and a *proposed* window only; no ALLOW/BLOCK, no policy objects |
| 2 | Agent 5 is the only component creating/applying policies | PASS | the only `policies.append` / `check()` / `consider()` are in `adaptive_defense.py`, called from `level2.replay_loop` |
| 3 | Replay uses `seq` order | PASS | input sorted by `seq`; Agent 4 rejects out-of-order/duplicates; trace strictly ascending; shuffled input → identical CSVs (test) |
| 4 | Activation exclusive, expiry inclusive | PASS | P0001: trigger F000014 = ALLOW, next match F000018 = BLOCK; F000114 (expiry) = BLOCK P0001; F000115 (after expiry) = ALLOW |
| 5 | Active policies checked before analysing a flow | PASS | `check()` runs first; all 98 blocked flows skipped Agent 4; all 102 allowed flows were analysed |
| 6 | `SRC_DST_PAIR` = exact `source\|destination` | PASS | `172.16.0.1\|192.168.10.50`; validator checks it is an observed pair |
| 7 | Why both policies use the same pair | — | Each policy was triggered by the first flow of a new group (G001 at F000014, G002 at F000115). A group of one flow has 1 destination and 1 port → narrowest scope (pair). Later flows from that source were blocked, so they never reached Agent 4 and the group never grew. |
| 8 | Was the source-wide (SOURCE_IP) condition ever triggered? | No | 0 SOURCE_IP policies. Effect is identical here: all 100 flows from 172.16.0.1 go to 192.168.10.50. |
| 9 | Why F000014 and F000115 were allowed | — | **F000014**: first suspicious flow of the replay (A1 BRUTE_FORCE 0.999999; A2 score 0.980, flagged anomalous; A3 attack_agreement); no policy existed yet → ALLOW, analysed → MEDIUM → P0001 for F000015–F000114. **F000115**: first record after P0001's inclusive expiry (F000114); no active policy → ALLOW, analysed (A1 DOS 1.0; A2 0.988, flagged) → P0002 for F000116–F000200. A policy can never block its own trigger. All 48 flows from that source between them were blocked by P0001. |
| 10 | No future records or hidden labels | PASS | hidden set has no labels; no label access in Level 2 code; prefix-replay test proves decisions on record k ignore records > k; no hardcoded IPs/IDs; IP-renaming test gives identical decisions |
| 11 | CSVs match the official portal schema | PASS | headers byte-identical to the portal templates; 200 / 200 / 2 rows; no CRLF, BOM or quoting; offline validator (portal rules, replay recomputed from the ledger): VALID. *The official portal is the final judge.* |
| 12 | Tests | PASS | **152 passed** (Agents 1–3: 133, Level 2: 19) |

**Terminology fixed (text only):** Agent 2 is an anomaly detector — it never classifies attacks. "Strong"
evidence now reads "Agent 1 predicted an attack and Agent 2 independently flagged it as anomalous" in Agent 4's
docstring/evidence text, Agent 3's rationale sentences and the README.
