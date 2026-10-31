"""Agent 3 - Evidence Fusion (CyberSentinel).

Combines Agent 1 (5-class LightGBM classifier) and Agent 2 (benign-only anomaly
detector) into one explainable, simulated ALLOW/BLOCK decision per flow.

    from src.agents.evidence_fusion import fuse, fuse_batch
    result = fuse(agent1_output, agent2_output, record_id="C000003")

Pure and deterministic: no model loading, no randomness, inputs are never
mutated, identical inputs give identical outputs. Agent outputs (or the
exception an agent raised) are validated strictly before any rule runs.

Decision rules, applied in order (rule id is returned in ``rule``):

  invalid_threat_evidence   Agent 1 output malformed/missing/raised -> UNKNOWN, BLOCK
  invalid_anomaly_evidence  Agent 2 output malformed/missing/raised -> Agent 1 class, BLOCK
  unreliable_anomaly        Agent 2 data_quality.reliable is false   -> UNCERTAIN, BLOCK
  low_confidence            Agent 1 confidence < min_confidence      -> UNCERTAIN, BLOCK
  attack_agreement          attack class + Agent 2 anomalous         -> AGREEMENT, BLOCK
  attack_over_normal        attack class + Agent 2 normal            -> DISAGREEMENT, BLOCK
  benign_over_anomaly       BENIGN + Agent 2 anomalous               -> DISAGREEMENT, ALLOW (review)
  benign_agreement          BENIGN + Agent 2 normal                  -> AGREEMENT, ALLOW
  bayes_escalation          (experimental, OFF by default; see FusionConfig)

Principles: Agent 2 can never supply or change an attack class (it has none);
an Agent 2 "normal" never downgrades an Agent 1 attack; invalid evidence never
yields ALLOW. ``anomaly_score`` is Agent 2's benign-unusualness percentile
(0.97 = more unusual than 97% of held-out normal flows), NOT an attack
probability, and is never described as one.
"""
from __future__ import annotations

import logging
import math
import numbers
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
AGENT_NAME = "evidence_fusion"
CLASSES = ("BENIGN", "BRUTE_FORCE", "DOS", "DDOS", "PORT_SCAN")
ATTACKS = CLASSES[1:]
UNKNOWN = "UNKNOWN"
ALLOW, BLOCK = "ALLOW", "BLOCK"
AGREEMENT, DISAGREEMENT = "AGREEMENT", "DISAGREEMENT"
UNCERTAIN, INVALID_EVIDENCE = "UNCERTAIN", "INVALID_EVIDENCE"


@dataclass(frozen=True)
class FusionConfig:
    """Every tunable value of Agent 3. None was fitted on official eval labels.

    min_confidence: Agent 1 max-probability below this -> UNCERTAIN. 0.99 is a
        1% risk budget. Training holdout (36,021 rows neither agent fitted):
        conf < 0.99 held 2 rows / 1 error; conf >= 0.99 held 1 error in 36,019.
    benign_anomaly_decision: confident BENIGN + Agent 2 anomalous -> ALLOW
        (+ needs_review). Training-only 5-fold out-of-fold check (248,888 rows):
        1,378 such rows, 1,375 benign / 3 attacks. ALLOW costs 3 missed attacks;
        BLOCK would block ~3% of all benign traffic (1,375 rows). Not a guarantee.
    uncertain_decision / invalid_decision: fail-safe policy for UNCERTAIN /
        INVALID_EVIDENCE when the final class is not an attack.
    prob_sum_tolerance: Agent 1 rounds probabilities to 6 decimals; 1e-3 allows
        rounding but rejects genuinely unnormalised distributions.
    bayes_escalation (EXPERIMENTAL, default False): if Agent 1 says BENIGN but
        Agent 2 flags the flow, posterior odds of attack = Agent 1 attack odds x
        a2_likelihood_ratio; >= 0.5 switches to Agent 1's most likely *attack*
        class. Assumes the agents' errors are conditionally independent given
        the true class, which is NOT established (both read the same flow).
        a2_likelihood_ratio = recall / benign FPR at Agent 2's threshold on
        training data Agent 2 never fitted = 0.9924 / 0.0300 (= 33.1; 32.5 on the
        smaller holdout). It needs Agent 1 P(attack) >= ~0.03, i.e. confidence
        < 0.99, so it can only relabel flows already BLOCKed as UNCERTAIN; it never
        changes ALLOW/BLOCK at the default min_confidence. Training-only
        out-of-fold validation: fired on 1 of 248,888 rows (correct) - too
        little evidence to enable.
    """

    min_confidence: float = 0.99
    benign_anomaly_decision: str = ALLOW
    uncertain_decision: str = BLOCK
    invalid_decision: str = BLOCK
    prob_sum_tolerance: float = 1e-3
    confidence_tolerance: float = 1e-5
    bayes_escalation: bool = False
    a2_likelihood_ratio: float = 32.5
    max_evidence_features: int = 3

    def __post_init__(self):
        for name in ("benign_anomaly_decision", "uncertain_decision", "invalid_decision"):
            if getattr(self, name) not in (ALLOW, BLOCK):
                raise ValueError(f"{name} must be ALLOW or BLOCK, got {getattr(self, name)!r}")
        if not 0.0 < self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0, 1]")
        if not (_real(self.a2_likelihood_ratio) and self.a2_likelihood_ratio > 0):
            raise ValueError("a2_likelihood_ratio must be a positive finite number")



def _real(x: Any) -> bool:
    """Finite real number; bools and numeric strings are rejected."""
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)


DEFAULT_CONFIG = FusionConfig()


def _failed(out: Any, name: str) -> list[str] | None:
    if out is None:
        return [f"{name}: no output"]
    if isinstance(out, BaseException):
        return [f"{name}: raised {type(out).__name__}: {out}"]
    if not isinstance(out, Mapping):
        return [f"{name}: output is {type(out).__name__}, expected a mapping"]
    return None


def validate_threat(a1: Any, cfg: FusionConfig = DEFAULT_CONFIG) -> list[str]:
    """Problems with an Agent 1 output; empty list means valid."""
    if (p := _failed(a1, "agent1")) is not None:
        return p
    issues = []
    cls, conf, probs = a1.get("predicted_class"), a1.get("confidence"), a1.get("probabilities")
    if not isinstance(cls, str) or cls not in CLASSES:
        issues.append(f"agent1: unknown predicted_class {cls!r}")
    if not _real(conf) or not 0.0 <= conf <= 1.0:
        issues.append(f"agent1: confidence {conf!r} is not a finite number in [0, 1]")
    if not isinstance(probs, Mapping) or set(probs) != set(CLASSES):
        issues.append(f"agent1: probabilities must map exactly {list(CLASSES)}")
        return issues
    bad = [c for c in CLASSES if not _real(probs[c]) or not 0.0 <= probs[c] <= 1.0]
    if bad:
        issues.append(f"agent1: probabilities not finite numbers in [0, 1] for {bad}")
        return issues
    total = sum(probs[c] for c in CLASSES)
    if abs(total - 1.0) > cfg.prob_sum_tolerance:
        issues.append(f"agent1: probabilities sum to {total:.6f}, not 1")
    top = max(probs[c] for c in CLASSES)
    if not issues and probs[cls] != top:
        issues.append(f"agent1: predicted_class {cls} is not the most probable class")
    if not issues and abs(conf - top) > cfg.confidence_tolerance:
        issues.append(f"agent1: confidence {conf} does not equal the top probability {top}")
    return issues


def validate_anomaly(a2: Any, record_id: str | None = None) -> list[str]:
    """Problems with an Agent 2 output; empty list means valid."""
    if (p := _failed(a2, "agent2")) is not None:
        return p
    issues = []
    score, flag, thr = a2.get("anomaly_score"), a2.get("is_anomaly"), a2.get("threshold")
    dq = a2.get("data_quality")
    if not _real(score) or not 0.0 <= score <= 1.0:
        issues.append(f"agent2: anomaly_score {score!r} is not a finite number in [0, 1]")
    if not isinstance(flag, bool):
        issues.append(f"agent2: is_anomaly {flag!r} is not a boolean")
    if not _real(thr) or not 0.0 <= thr <= 1.0:
        issues.append(f"agent2: threshold {thr!r} is not a finite number in [0, 1]")
    if not issues and flag != (score >= thr):
        issues.append(f"agent2: is_anomaly={flag} contradicts score {score} vs threshold {thr}")
    if not isinstance(dq, Mapping) or not isinstance(dq.get("reliable"), bool):
        issues.append("agent2: data_quality.reliable missing or not a boolean")
    if not isinstance(a2.get("evidence", []), list):
        issues.append("agent2: evidence is not a list")
    rid = a2.get("record_id")
    if record_id is not None and rid is not None and str(rid) != str(record_id):
        issues.append(f"agent2: record_id {rid!r} does not match {record_id!r}")
    return issues


def _evidence_features(a2: Mapping, k: int) -> list[str]:
    feats = [e["feature"] for e in a2.get("evidence", [])
             if isinstance(e, Mapping) and isinstance(e.get("feature"), str)]
    return feats[:k]


def _a1_text(a1: Mapping) -> str:
    return f"Agent 1 classified the flow as {a1['predicted_class']} (confidence {a1['confidence']:.4f})."


def _a2_text(a2: Mapping, cfg: FusionConfig) -> str:
    s = a2["anomaly_score"]
    text = (f"Agent 2 gave it an unusualness score of {s:.3f} (more unusual than {s * 100:.1f}% of "
            f"held-out normal flows; an unusualness percentile, not an attack probability) and "
            f"{'flagged it as anomalous' if a2['is_anomaly'] else 'did not flag it'} "
            f"(threshold {a2['threshold']:.3f}).")
    feats = _evidence_features(a2, cfg.max_evidence_features)
    if feats:
        text += f" Main deviating features: {', '.join(feats)}."
    return text


def _result(record_id, final_class, decision, status, rule, rationale, issues, a1, a2, cfg,
            a1_ok=True, a2_ok=True):
    return {
        "agent": AGENT_NAME,
        "schema_version": SCHEMA_VERSION,
        "record_id": None if record_id is None else str(record_id),
        "final_class": final_class,
        "decision": decision,
        "evidence_status": status,
        "needs_review": status != AGREEMENT,
        "rule": rule,
        "rationale": rationale,
        "evidence_issues": issues,
        "signals": {
            "threat_class": a1["predicted_class"] if a1_ok else None,
            "threat_confidence": a1["confidence"] if a1_ok else None,
            "anomaly_score": a2["anomaly_score"] if a2_ok else None,
            "is_anomaly": a2["is_anomaly"] if a2_ok else None,
            "anomaly_reliable": a2["data_quality"]["reliable"] if a2_ok else None,
            "anomaly_features": _evidence_features(a2, cfg.max_evidence_features) if a2_ok else [],
        },
    }


def fuse(a1: Any, a2: Any, cfg: FusionConfig = DEFAULT_CONFIG, record_id: str | None = None) -> dict:
    """Fuse one flow's Agent 1 and Agent 2 outputs (or the exceptions they raised)."""
    if record_id is None and isinstance(a2, Mapping) and isinstance(a2.get("record_id"), str):
        record_id = a2["record_id"]
    p1, p2 = validate_threat(a1, cfg), validate_anomaly(a2, record_id)
    if p1 or p2:
        log.warning("fusion.invalid_evidence record_id=%s issues=%s", record_id, p1 + p2)
        if p1:
            return _result(record_id, UNKNOWN, cfg.invalid_decision, INVALID_EVIDENCE,
                           "invalid_threat_evidence",
                           "Agent 1's output failed validation, so the flow cannot be classified "
                           f"and is handled fail-safe ({cfg.invalid_decision}): " + "; ".join(p1 + p2) + ".",
                           p1 + p2, a1, a2, cfg, a1_ok=False, a2_ok=not p2)
        cls = a1["predicted_class"]
        decision = BLOCK if cls in ATTACKS else cfg.invalid_decision
        return _result(record_id, cls, decision, INVALID_EVIDENCE, "invalid_anomaly_evidence",
                       f"{_a1_text(a1)} Agent 2's output failed validation ({'; '.join(p2)}), so the "
                       f"classification could not be cross-checked; handled fail-safe ({decision}).",
                       p2, a1, a2, cfg, a2_ok=False)

    cls, conf, probs = a1["predicted_class"], a1["confidence"], a1["probabilities"]
    flag, reliable = a2["is_anomaly"], a2["data_quality"]["reliable"]
    evidence = f"{_a1_text(a1)} {_a2_text(a2, cfg)}"
    final, escalated = cls, False
    if cfg.bayes_escalation and cls == "BENIGN" and flag and reliable:
        q = 1.0 - probs["BENIGN"]
        odds = q / max(1.0 - q, 1e-12) * cfg.a2_likelihood_ratio
        if odds / (1.0 + odds) >= 0.5:
            final, escalated = max(ATTACKS, key=lambda c: (probs[c], -ATTACKS.index(c))), True

    def out(decision, status, rule, conclusion):
        return _result(record_id, final, decision, status, rule, f"{evidence} {conclusion}", [], a1, a2, cfg)

    attack = final in ATTACKS
    if not reliable:
        imputed = a2["data_quality"].get("imputed_model_features") or []
        d = BLOCK if attack else cfg.uncertain_decision
        return out(d, UNCERTAIN, "unreliable_anomaly",
                   f"Agent 2 marked its own result unreliable ({len(imputed)} model features had to be "
                   f"imputed), so the evidence is insufficient; handled fail-safe ({d}).")
    if conf < cfg.min_confidence:
        d = BLOCK if attack else cfg.uncertain_decision
        return out(d, UNCERTAIN, "low_confidence",
                   f"Agent 1's confidence is below the {cfg.min_confidence} floor, so the classification "
                   f"is uncertain; handled fail-safe ({d}).")
    if escalated:
        return out(BLOCK, DISAGREEMENT, "bayes_escalation",
                   f"EXPERIMENTAL rule: Agent 1's residual attack probability combined with Agent 2's "
                   f"anomaly flag makes an attack more likely than not; final class {final} is Agent 1's "
                   f"most probable attack class.")
    if attack and flag:
        return out(BLOCK, AGREEMENT, "attack_agreement",
                   "Agent 1 classifies it as an attack and Agent 2 independently flags it as anomalous: BLOCK.")
    if attack:
        return out(BLOCK, DISAGREEMENT, "attack_over_normal",
                   f"Agent 2 did not flag it, but an anomaly detector's 'normal' does not rule out an "
                   f"attack (it misses some slow/low-volume attacks), so Agent 1's confident {final} "
                   f"classification stands: BLOCK.")
    if flag:
        d = cfg.benign_anomaly_decision
        return out(d, DISAGREEMENT, "benign_over_anomaly",
                   f"Agent 1 is confident the flow is benign, but Agent 2 finds it unusual; the agents "
                   f"disagree. Policy: {d}, flagged for analyst review.")
    return out(ALLOW, AGREEMENT, "benign_agreement", "Agent 1 classifies it as benign and Agent 2 does not flag it as anomalous: ALLOW.")


def fuse_batch(a1s: Sequence, a2s: Sequence, record_ids: Sequence | None = None,
               cfg: FusionConfig = DEFAULT_CONFIG) -> list[dict]:
    """Fuse aligned lists of agent outputs. Lengths must match exactly."""
    if len(a1s) != len(a2s) or (record_ids is not None and len(record_ids) != len(a1s)):
        raise ValueError(f"misaligned batch: {len(a1s)} agent1, {len(a2s)} agent2, "
                         f"{'-' if record_ids is None else len(record_ids)} record_ids")
    ids = record_ids if record_ids is not None else [None] * len(a1s)
    return [fuse(a, b, cfg, r) for a, b, r in zip(a1s, a2s, ids)]
