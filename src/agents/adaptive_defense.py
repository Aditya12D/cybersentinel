"""Agent 5 - Adaptive Defense (CyberSentinel Level 2): temporary, simulated BLOCK policies.

Replay contract (identical to the organiser validator):

    for each record in seq order:
        action, policy_id = defense.check(flow)          # earlier policies are applied first
        if action == "ALLOW":
            corr = agent4.observe(flow, agent3_result)   # analyse only allowed flows
            defense.consider(flow, corr)                 # may create a policy for LATER records

A policy created at record i activates on record i+1 (activation exclusive) and stays active
through its expiry record (inclusive). A BLOCK reports the lexicographically smallest matching
policy_id; an ALLOW reports NONE. Scope values are exact observed endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

SOURCE_IP, DESTINATION_IP, SRC_DST_PAIR = "SOURCE_IP", "DESTINATION_IP", "SRC_DST_PAIR"


@dataclass(frozen=True)
class DefenseConfig:
    """ttl_records: policy lifetime in records after its trigger (100 mirrors the organiser's
    example P0001: F000012 -> F000112). trigger_strengths: Agent 4 evidence levels that justify
    a policy - a single LOW (weak, uncorroborated) flow never blocks anyone."""

    ttl_records: int = 100
    trigger_strengths: tuple = ("HIGH", "MEDIUM")


def scope_matches(scope_type: str, scope_value: str, flow: Mapping) -> bool:
    src, dst = flow.get("source_ip"), flow.get("destination_ip")
    return ((scope_type == SOURCE_IP and scope_value == src)
            or (scope_type == DESTINATION_IP and scope_value == dst)
            or (scope_type == SRC_DST_PAIR and scope_value == f"{src}|{dst}"))


class AdaptiveDefense:
    def __init__(self, record_ids_in_seq_order: Sequence[str], cfg: DefenseConfig = DefenseConfig()):
        self.cfg = cfg
        self.order = list(record_ids_in_seq_order)
        self.pos = {rid: i for i, rid in enumerate(self.order, start=1)}
        if len(self.pos) != len(self.order):
            raise ValueError("duplicate record_id in replay order")
        self.policies: list[dict] = []

    def _active(self, rid: str, flow: Mapping) -> list[str]:
        i = self.pos[rid]
        return sorted(p["policy_id"] for p in self.policies
                      if p["_act"] < i <= p["_exp"] and scope_matches(p["scope_type"], p["scope_value"], flow))

    def check(self, flow: Mapping) -> tuple[str, str]:
        m = self._active(str(flow["record_id"]), flow)
        return ("BLOCK", m[0]) if m else ("ALLOW", "NONE")

    def consider(self, flow: Mapping, corr: Mapping) -> dict | None:
        """Create a policy from Agent 4's evidence for the current (allowed) flow, or None."""
        if corr.get("evidence_strength") not in self.cfg.trigger_strengths:
            return None
        g = corr["group"]
        rid = str(flow["record_id"])
        src, dst, i = flow["source_ip"], flow["destination_ip"], self.pos[rid]
        exp = min(i + self.cfg.ttl_records, len(self.order))
        if exp <= i:
            return None                                   # nothing later left to protect
        if g["unique_destinations"] >= 2 or g["unique_destination_ports"] >= 2:
            scope_type, value = SOURCE_IP, src            # source touching several services/hosts
        else:
            scope_type, value = SRC_DST_PAIR, f"{src}|{dst}"   # narrowest scope, preserves other traffic
        p = {"policy_id": f"P{len(self.policies) + 1:04d}", "activated_after_record_id": rid,
             "scope_type": scope_type, "scope_value": value, "expires_after_record_id": self.order[exp - 1],
             "policy_action": "BLOCK", "_act": i, "_exp": exp, "group_id": g["group_id"],
             "evidence_strength": corr["evidence_strength"], "pattern": g["pattern"], "evidence": g["evidence"]}
        self.policies.append(p)
        return p

    def ledger(self) -> list[dict]:
        return [{k: v for k, v in p.items() if not k.startswith("_")} for p in self.policies]
