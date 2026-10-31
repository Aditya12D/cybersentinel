"""Agent 4 - Temporal Correlation (CyberSentinel Level 2).

Groups related suspicious flows from the same source, in replay order, using observed
endpoints and a time window. It never decides ALLOW/BLOCK; Agent 5 does.

    corr = TemporalCorrelator()
    for flow, a3 in replay:                     # strictly ascending seq
        result = corr.observe(flow, a3)         # uses this flow + earlier flows only

Replay safety: ``observe`` rejects out-of-order or duplicate records, keeps only past
state, and never reads labels. A flow is *suspicious* when Agent 3 recommends BLOCK;
it is *strong* when Agent 1 predicts an attack class and Agent 2 independently flags the flow as
anomalous (Agent 3 rule attack_agreement). Agent 2 never classifies attacks; it measures unusualness.

Per suspicious flow it also proposes the policy window (portal: A4 outputs activation and expiry
index): activation_index = this record's seq (a policy may only start on the next record),
expiry_index = seq + policy_window_records. Agent 5 decides whether to enact it and with what scope.
Destination-side correlation: destination_fan_in = distinct suspicious sources that hit the same
destination within the window (MULTI_SOURCE_TARGET when >= 3, e.g. distributed attacks).

Evidence strength of the group a suspicious flow joins:
  HIGH    >= min_group_flows suspicious flows in the window, at least one strong
  MEDIUM  one strong flow, or >= min_group_flows weak flows
  LOW     a single weak flow (e.g. low-confidence / unreliable evidence)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping

STRONG_RULES = {"attack_agreement"}


@dataclass(frozen=True)
class CorrelationConfig:
    """window_seconds: max gap between consecutive suspicious flows of one source for them to
    stay in the same group. 300 s is an a-priori choice (a scan / brute-force / DoS burst is
    seconds-to-minutes); it could not be validated on labelled data because the training pack
    has no IPs or timestamps.
    min_group_flows: suspicious flows needed before weak evidence alone counts as MEDIUM.
    policy_window_records: length of the proposed policy window in records (100 mirrors the
    organiser's example P0001: F000012 -> F000112).
    multi_source_min: distinct suspicious sources on one destination to flag MULTI_SOURCE_TARGET."""

    window_seconds: float = 300.0
    min_group_flows: int = 2
    policy_window_records: int = 100
    multi_source_min: int = 3


def flow_timestamp(day, time_offset) -> datetime | None:
    """day 'YYYY-MM-DD' + time_offset '[HH:]MM:SS[.f]' -> datetime; None if unparseable.
    Never compares strings: different days are different dates."""
    try:
        parts = [float(p) for p in str(time_offset).split(":")]
        if not 2 <= len(parts) <= 3:
            return None
        secs = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
        return datetime.strptime(str(day), "%Y-%m-%d") + timedelta(seconds=secs)
    except (TypeError, ValueError):
        return None


class TemporalCorrelator:
    def __init__(self, cfg: CorrelationConfig = CorrelationConfig()):
        self.cfg = cfg
        self.last_seq = 0
        self.seen: set[str] = set()
        self.open: dict[str, dict] = {}       # source_ip -> current group
        self.targets: dict[str, list] = {}    # destination_ip -> recent (timestamp, source_ip)
        self.groups: list[dict] = []          # every group ever formed (for logs / demo)

    def observe(self, flow: Mapping, a3: Mapping) -> dict:
        seq, rid = int(flow["seq"]), str(flow["record_id"])
        if seq <= self.last_seq:
            raise ValueError(f"out-of-order record {rid}: seq {seq} after {self.last_seq}")
        if rid in self.seen:
            raise ValueError(f"duplicate record_id {rid}")
        self.last_seq = seq
        self.seen.add(rid)
        src, dst = flow.get("source_ip"), flow.get("destination_ip")
        base = {"record_id": rid, "seq": seq, "suspicious": a3.get("decision") == "BLOCK",
                "group_id": None, "evidence_strength": None, "group": None}
        if not base["suspicious"]:
            return base
        if not isinstance(src, str) or not src or not isinstance(dst, str) or not dst:
            return {**base, "note": "missing endpoint: cannot correlate"}

        ts = flow_timestamp(flow.get("day"), flow.get("time_offset"))
        g = self.open.get(src)
        gap = None if g is None or ts is None or g["_last_ts"] is None else (ts - g["_last_ts"]).total_seconds()
        if g is None or gap is None or gap > self.cfg.window_seconds or gap < 0:
            g = {"group_id": f"G{len(self.groups) + 1:03d}", "source_ip": src, "member_records": [],
                 "first_seq": seq, "_first_ts": ts, "_dsts": Counter(), "_ports": set(), "_protos": set(),
                 "_classes": Counter(), "_strong": 0}
            self.groups.append(g)
            self.open[src] = g
        g["member_records"].append(rid)
        g["last_seq"], g["_last_ts"] = seq, ts
        g["_dsts"][dst] += 1
        g["_ports"].add(flow.get("destination_port"))
        g["_protos"].add(flow.get("protocol"))
        g["_classes"][a3.get("final_class")] += 1
        g["_strong"] += a3.get("rule") in STRONG_RULES
        snap = self._snapshot(g)
        recent = [(t, sip) for t, sip in self.targets.get(dst, [])
                  if t is not None and ts is not None and (ts - t).total_seconds() <= self.cfg.window_seconds]
        self.targets[dst] = recent + [(ts, src)]
        fan_in = len({sip for _, sip in self.targets[dst]})
        return {**base, "group_id": g["group_id"], "evidence_strength": snap["evidence_strength"], "group": snap,
                "activation_index": seq, "expiry_index": seq + self.cfg.policy_window_records,
                "destination_fan_in": fan_in,
                "destination_pattern": "MULTI_SOURCE_TARGET" if fan_in >= self.cfg.multi_source_min else None}

    def _snapshot(self, g: dict) -> dict:
        n, strong = len(g["member_records"]), g["_strong"]
        if n >= self.cfg.min_group_flows and strong:
            strength = "HIGH"
        elif strong or n >= self.cfg.min_group_flows:
            strength = "MEDIUM"
        else:
            strength = "LOW"
        ports, dsts = len(g["_ports"]), len(g["_dsts"])
        pattern = ("MULTI_PORT_ACTIVITY" if ports >= 3 else "MULTI_DESTINATION" if dsts >= 3
                   else "REPEATED_PAIR" if n >= 2 else "SINGLE_FLOW")
        span = (g["_last_ts"] - g["_first_ts"]).total_seconds() if g["_first_ts"] and g["_last_ts"] else None
        top_dst = g["_dsts"].most_common(1)[0][0]
        return {
            "group_id": g["group_id"], "source_ip": g["source_ip"],
            "destination_ip": top_dst if dsts == 1 else None,
            "member_records": list(g["member_records"]), "first_seq": g["first_seq"], "last_seq": g["last_seq"],
            "suspicious_flow_count": n, "strong_flow_count": strong,
            "unique_destinations": dsts, "unique_destination_ports": ports,
            "protocols": sorted(p for p in g["_protos"] if p is not None),
            "agent3_classes": dict(g["_classes"]), "time_window_seconds": span,
            "pattern": pattern, "evidence_strength": strength,
            "evidence": (f"{n} suspicious flow(s) from {g['source_ip']} ({strong} where Agent 1 predicted an attack and Agent 2 flagged it anomalous) to "
                         f"{dsts} destination(s), {ports} port(s), seq {g['first_seq']}-{g['last_seq']}"
                         + (f" over {span:.0f} s" if span is not None else "")),
        }

    def report(self) -> list[dict]:
        return [self._snapshot(g) for g in self.groups]
