# CyberSentinel — Panel Demo Script

Run everything from the repo root. `python demo.py` opens a menu (1–8); `python demo.py <n>` runs one
section; `python demo.py all` runs them all (~6 s). Every number on screen is computed live by the real
agents on the 200 hidden records — nothing is hardcoded. Sections are numbered as in the menu.

## The five agents in one sentence each

| Agent | Question it answers | Output |
|---|---|---|
| 1 Threat Classification | What type of traffic is this flow most likely? | class + confidence + 5 probabilities |
| 2 Anomaly Detection | How unusual is it compared with normal traffic? | anomaly score (percentile of normal), flag, feature evidence, data quality |
| 3 Evidence Fusion | Do the two independent signals support each other? | final class, AGREE / CONFLICT / UNCERTAIN, initial decision, rationale |
| 4 Temporal Correlation | Which suspicious flows belong together over time? | groups, evidence strength, proposed policy window |
| 5 Adaptive Defense | Should later traffic be blocked, and how narrowly? | temporary policies, ALLOW/BLOCK per flow |

Say it precisely: *"Agent 1 predicted an attack and Agent 2 independently flagged it as anomalous."*
Agent 2 never names an attack type.

## 5-minute version

| Time | Say | Show |
|---|---|---|
| 0:00–0:40 | "CyberSentinel is a five-agent intrusion detection and adaptive defence system. Two independent detectors, a fusion agent, a temporal correlator, and a defence agent that issues temporary, explainable policies during a chronological replay." | Section 6 diagram |
| 0:40–1:30 | "Agent 1 is a LightGBM classifier on 71 flow features — 7 zero-information columns removed. Agent 2 is trained only on normal traffic: kNN distance plus Isolation Forest, score = how unusual vs normal, not an attack probability." | Sections 1 and 2 tables |
| 1:30–2:15 | "Agent 3 fuses them with ordered rules. If they agree we act; if they disagree we flag for review; invalid or unreliable evidence never produces ALLOW." | Section 3 rule table + examples |
| 2:15–3:30 | "In replay order, Agent 5 checks active policies first. F000014 is the first suspicious flow — Agent 1 predicted BRUTE_FORCE at 0.999999 and Agent 2 independently flagged it (0.980). No policy existed, so it is allowed, Agent 4 groups it, and Agent 5 creates P0001 for later records only. F000018 is then blocked by P0001 before analysis." | Section 7 (P0001 trigger, first block) |
| 3:30–4:15 | "Policies are temporary: P0001 still blocks its expiry record F000114; F000115 arrives after expiry, is analysed again, and creates P0002." | Section 7 (last block, after expiry) |
| 4:15–5:00 | "Result: 200 records, 98 blocked by 2 policies, 0 flows predicted benign were blocked, and the live run reproduces our submitted CSVs byte-for-byte. These are predictions — the hidden set has no labels." | Section 8 + IDENTICAL check in section 6 |

## 10-minute version

1. **Architecture (1 min)** — Section 6 diagram. Point out the real order: A1 and A2 run independently on each
   flow; A3 fuses; then the replay loop: *A5 check → (BLOCK, stop) or (ALLOW → A4 → A5 may create a policy)*.
2. **Agent 1 (1.5 min)** — Section 1. Input = flow statistics only (no IPs, no time, no labels). Show the five
   probabilities summing to 1. Day 1 result: eval macro-F1 0.99998.
3. **Agent 2 (1.5 min)** — Section 2. Why a second detector: it is trained on normal traffic only, so it can
   notice unfamiliar behaviour. Explain percentile score and the 0.97 = 3% false-alarm budget. Show feature
   evidence and its own explanation for F000014.
4. **Agent 3 (1.5 min)** — Section 3. Walk the 8 ordered rules. Real hidden examples (AGREE, CONFLICT), then
   validation-set examples for rules that don't occur in the hidden set (attack + normal; low confidence), and
   two clearly labelled TEST FIXTURES (half the features blanked → unreliable → BLOCK; Agent 2 crash → BLOCK).
5. **Agent 4 (1.5 min)** — Section 4. Timestamp = day + time_offset; 300 s window; strength levels; proposed
   window. Real replay forms only 2 groups because blocked flows are never analysed; the illustration (no
   blocking) shows groups growing to HIGH and MULTI_PORT_ACTIVITY for the port scan.
6. **Agent 5 (1.5 min)** — Section 5. Scopes; why narrow first; activation exclusive / expiry inclusive; ledger
   with 48 + 50 blocked records; fixture showing the smallest matching policy id rule.
7. **Lifecycle on real records (1 min)** — Section 7: trigger, first block, last block in window, after expiry.
   Each line also shows the frozen `replay.csv` value next to the live one.
8. **Results + honesty (0.5 min)** — Section 8; state the limitations below.

## Questions the panel may ask

**Why five agents?** Each answers a different question: what is it (A1), how unusual is it (A2), do the
signals agree (A3), is it part of a pattern over time (A4), what should we do about later traffic (A5). The
outputs feed each other: A3 decides which flows are suspicious for A4; A4's evidence and window drive A5;
A5's policies decide which flows are analysed at all.

**Why both classification and anomaly detection?** A classifier only recognises the classes it was trained on.
Our audit showed that with an attack type removed from training, all five classifier families gave it 0%
recall — at ≥ 0.99 confidence. Agent 2, trained only on normal traffic, still flagged 99.6% of that unseen
attack type. They fail in different ways, so together they are more trustworthy.

**What if Agents 1 and 2 disagree?** Attack + "normal" → still BLOCK (an anomaly detector misses some slow
attacks; it never downgrades a classifier attack). Confident benign + anomalous → ALLOW but flagged for review
(CONFLICT); training data showed 1,375 of 1,378 such flows were benign, so blocking them would block ~3% of
normal traffic.

**Why does Agent 4 not block?** Separation of duties: correlation is evidence, enforcement is policy. One
component (A5) owns every policy, so every BLOCK traces to exactly one ledger entry.

**Why is Agent 5 the only policy owner?** The portal recomputes every BLOCK from `policies.csv`; a single owner
keeps the ledger consistent and auditable.

**How do you prevent future-data leakage?** Strict `seq` order; Agent 4 rejects out-of-order or duplicate
records; a test replays every prefix of the data and proves decisions on record *k* never change when later
records are added; no labels exist in the hidden set; no IP or record ID is hardcoded (renaming every IP gives
identical decisions).

**Why is the triggering flow allowed?** The rules say a policy activates on the record *after* its trigger.
The trigger is analysed because no policy existed when it arrived.

**Why are policies temporary, and what happens at expiry?** A source can stop being malicious, and permanent
blocks destroy legitimate traffic. After expiry the next flow is analysed from scratch; if the evidence is still
there, a new policy is created (P0002). Cost: one re-trigger flow per expiry.

**How do you avoid blocking all traffic from a source?** Agent 5 starts with the narrowest scope
(source|destination pair) and widens to the whole source only when that source has touched several ports or
hosts. DESTINATION_IP is never used — it would block every legitimate client of a server.

**How do you know the model generalises?** Honestly: on random splits of the same captures it is near-perfect
(eval macro-F1 0.99998; duplicate-free 0.99997). Our own audit with capture-order (time-like) splits showed it
degrades on new time windows (DOS recall 0.96–0.98) and cannot recognise an attack type it never saw. That is
precisely why Agent 2 and the fail-safe fusion rules exist. Details: `docs/audit/`.

## Limitations to state openly

- Hidden-set results are predictions; without labels we report no accuracy, FP or FN rates for Level 2.
- The 300 s window and 100-record policy lifetime could not be validated on labelled data (the training
  pack has no IPs or timestamps); the replay is identical for windows of 60–3600 s.
- The first flow of every campaign is always allowed by design.
- Agent 2's false-alarm rate is higher on web traffic (≈ 8% on port 443 vs 3% overall).
- Eval labels influenced two Day 1 choices (Agent 1 model selection among near-identical candidates; Agent 2's
  threshold); both were re-justified with training data only (`docs/audit/EVALUATION_INTEGRITY.md`).
