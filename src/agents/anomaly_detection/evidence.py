"""Feature-level evidence and rationale for Agent 2.

Two measured quantities per feature; nothing is hand-written per attack type.

1. kNN contribution (model-grounded). The kNN member scores a flow by its mean
   Euclidean distance to the k closest benign prototypes in standardised log
   space. The squared distance is a sum over features, so
       share_f = mean_k (z_f - n_kf)^2 / sum_f mean_k (z_f - n_kf)^2
   is an exact additive decomposition: which features separate this flow from
   the most similar normal flows, and by how much.

2. Benign tail probability (distribution-grounded). For the observed raw value
   x, using the held-out BENIGN reference set (never used for fitting):
       upper = P_benign(X >= x),  lower = P_benign(X <= x)
       tail  = min(upper, lower), direction = "above" if upper < lower else "below"
   Severity is derived only from ``tail`` (config.SEVERITY_TAIL_PROBABILITY):
       high <= 0.001 < medium <= 0.01 < low <= 0.05 < info
   "info" marks a feature whose value is individually common among benign
   flows but which still separates the flow from its nearest benign neighbours
   (i.e. the anomaly is in the *combination* of values).

Evidence selection: model features ranked by kNN share, taken until 80% of the
distance is explained (max ``EVIDENCE_TOP_K``, min share 5%); remaining slots
are filled with the rarest marginal values (severity medium or high). The
model features are already de-correlated (|r| < 0.95), so evidence items do not
repeat the same behaviour under different names. Features whose value was
imputed (missing / invalid / non-finite input) are never reported as evidence;
they are listed in ``data_quality`` instead.
"""
from __future__ import annotations

import math

import numpy as np

from . import config
from .detector import AnomalyDetector, ScoreResult

_SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1, "info": 0}
MIN_SHARE = 0.05
CUMULATIVE_SHARE = 0.80

# Feature-name keyword -> behaviour category used in the rationale text. Order
# matters: the first matching rule wins.
_CATEGORY_RULES = (
    ("Packets/s", "packet rate"),
    ("Bytes/s", "byte rate"),
    ("Bulk Rate", "bulk transfer rate"),
    ("IAT", "packet inter-arrival time"),
    ("Flow Duration", "flow duration"),
    ("Active", "active-period duration"),
    ("Idle", "idle-period duration"),
    ("Init_Win", "TCP initial window size"),
    ("Flag", "TCP flag usage"),
    ("Header Length", "header length"),
    ("Down/Up", "download/upload ratio"),
    ("Bulk", "bulk transfer size"),
    ("Length", "packet size"),
    ("Size", "packet size"),
    ("Bytes", "byte volume"),
    ("Packets", "packet count"),
    ("act_data_pkt", "data packet count"),
    ("min_seg_size", "minimum segment size"),
)


def feature_category(name: str) -> str:
    for key, cat in _CATEGORY_RULES:
        if key in name:
            return cat
    return name


def severity_from_tail(tail: float) -> str:
    for level, limit in config.SEVERITY_TAIL_PROBABILITY:
        if tail <= limit:
            return level
    return "info"


def _inverse_raw(det: AnomalyDetector, z: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Map standardised log-space values back to raw units (for neighbour values)."""
    pre = det.preprocessor
    t = z * pre.log_scale_[cols] + pre.log_median_[cols]
    return np.sign(t) * np.expm1(np.abs(t))


def _fmt(v: float) -> str:
    if v == 0 or not math.isfinite(v):
        return f"{v:g}"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        return f"{v:.3g}"
    return f"{v:,.6g}" if abs(v) >= 1000 else f"{v:.4g}"


def _reference(det: AnomalyDetector):
    """Benign reference columns for the model features (sorted) and their
    1st/50th/99th percentiles. Cached on the detector (not pickled)."""
    hit = det.__dict__.get("_evidence_ref_")
    if hit is None:
        ref = np.ascontiguousarray(det.ref_raw_sorted_[:, np.asarray(det.model_idx_)].T)
        n_ref = ref.shape[1]
        qs = tuple(ref[:, int(round(q * (n_ref - 1)))] for q in (0.01, 0.5, 0.99))
        hit = (ref, qs)
        det._evidence_ref_ = hit
    return hit[0].T, hit[1]


def compute_evidence(det: AnomalyDetector, res: ScoreResult, is_anomaly: np.ndarray,
                     top_k: int = config.EVIDENCE_TOP_K) -> list[list[dict]]:
    """Evidence lists for every row of ``res``. Normal flows only report
    individually rare values (severity >= medium); anomalous flows report the
    full ranked evidence."""
    n = len(res.anomaly_score)
    cols = np.asarray(det.model_idx_)
    feats = det.features
    ref, (q01, q50, q99) = _reference(det)
    n_ref = ref.shape[0]
    raw = res.raw[:, cols]

    # Benign tail probabilities, vectorised per feature.
    upper = np.empty_like(raw)
    lower = np.empty_like(raw)
    ref_cols = ref.T  # (n_features, n_ref), each row contiguous
    for j in range(len(cols)):
        cj = ref_cols[j]
        x = raw[:, j].astype(np.float32)
        lower[:, j] = np.searchsorted(cj, x, side="right") / n_ref
        upper[:, j] = 1.0 - np.searchsorted(cj, x, side="left") / n_ref
    tail = np.minimum(upper, lower)
    # Imputed values (missing / invalid / non-finite input) are not observations
    # and must never be presented as evidence.
    imputed = (res.report.imputed[:, cols] if res.report.imputed is not None
               else np.zeros(raw.shape, dtype=bool))
    share_mask = np.where(imputed, 0.0, 1.0)
    tail = np.where(imputed, np.inf, tail)

    out: list[list[dict]] = []
    chunk = 10_000
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        zm = res.z[s:e][:, cols]
        nb = det.prototypes_[res.knn_neighbors[s:e]]            # (b, k, f)
        sq = ((zm[:, None, :] - nb) ** 2).mean(axis=1)          # (b, f)
        share = sq / np.maximum(sq.sum(axis=1, keepdims=True), 1e-12)
        share_ev = share * share_mask[s:e]   # true shares, imputed features zeroed
        nb_lo = _inverse_raw(det, nb.min(axis=1), cols)          # (b, f) raw units
        nb_hi = _inverse_raw(det, nb.max(axis=1), cols)
        for r in range(e - s):
            i = s + r
            out.append(_row_evidence(i, r, share_ev, nb_lo, nb_hi, raw, tail, upper, lower,
                                     q01, q50, q99, cols, feats, bool(is_anomaly[i]), top_k))
    return out


def _row_evidence(i, r, share, nb_lo, nb_hi, raw, tail, upper, lower, q01, q50, q99,
                  cols, feats, anomalous, top_k):
    nb_count = config.KNN_N_NEIGHBORS
    chosen: list[int] = []
    if anomalous:
        order = np.argsort(-share[r])
        cum = 0.0
        for j in order:
            if len(chosen) >= top_k or cum >= CUMULATIVE_SHARE or share[r, j] < MIN_SHARE:
                break
            chosen.append(int(j))
            cum += share[r, j]
    # Rarest marginal values (medium/high) fill remaining slots.
    for j in np.argsort(tail[i]):
        if len(chosen) >= top_k or tail[i, j] > 0.01:
            break
        if int(j) not in chosen:
            chosen.append(int(j))

    items = []
    for j in chosen:
        t = float(tail[i, j])
        sev = severity_from_tail(t)
        direction = "above" if upper[i, j] < lower[i, j] else "below"
        x = float(raw[i, j])
        # neighbours are stored as float32 in log space; trim round-trip noise
        lo, hi = float(f"{nb_lo[r, j]:.6g}"), float(f"{nb_hi[r, j]:.6g}")
        nb_txt = _fmt(lo) if _fmt(lo) == _fmt(hi) else f"{_fmt(lo)} to {_fmt(hi)}"
        if sev == "info":
            reference = (f"Within benign range (benign median {_fmt(float(q50[j]))}, "
                         f"1st-99th percentile {_fmt(float(q01[j]))}-{_fmt(float(q99[j]))}); "
                         f"accounts for {share[r, j] * 100:.0f}% of the distance to the "
                         f"{nb_count} most similar benign flows (their values: {nb_txt})")
        else:
            word = "high" if direction == "above" else "low"
            pct = (upper if direction == "above" else lower)[i, j] * 100
            reference = (f"{'Above' if direction == 'above' else 'Below'} benign reference range: "
                         f"only {pct:.2f}% of benign flows have a value this {word} or {'higher' if word == 'high' else 'lower'} "
                         f"(benign median {_fmt(float(q50[j]))}, "
                         f"1st-99th percentile {_fmt(float(q01[j]))}-{_fmt(float(q99[j]))})")
        items.append({
            "feature": feats[cols[j]],
            "observed_value": x,
            "reference": reference,
            "severity": sev,
            "direction": direction,
            "benign_tail_probability": round(t, 6),
            "benign_median": float(q50[j]),
            "benign_p01": float(q01[j]),
            "benign_p99": float(q99[j]),
            "nearest_benign_range": [lo, hi],
            "distance_contribution": round(float(share[r, j]), 4),
        })
    items.sort(key=lambda d: (-_SEVERITY_ORDER[d["severity"]], -d["distance_contribution"]))
    return items


def build_rationale(score: float, threshold: float, is_anomaly: bool, p_knn: float,
                    p_if: float, evidence: list[dict], n_imputed: int = 0) -> str:
    head = (f"Anomaly score {score:.3f}: more unusual than {score * 100:.1f}% of benign "
            f"reference flows (threshold {threshold:.3f}).")
    if n_imputed:
        head += (f" {n_imputed} model feature(s) were missing or invalid and replaced by "
                 f"benign reference values; see data_quality.")
    if not is_anomaly:
        msg = f"{head} The flow is consistent with normal benign traffic."
        if evidence:
            msg += " Individually rare values: " + _phrases(evidence) + "."
        return msg
    drivers = []
    if p_knn >= threshold:
        drivers.append(f"far from all similar benign flows (kNN distance at benign percentile {p_knn * 100:.1f})")
    if p_if >= threshold:
        drivers.append(f"globally isolated (Isolation Forest at benign percentile {p_if * 100:.1f})")
    if not drivers:
        drivers.append(f"jointly unusual on both detectors (kNN {p_knn * 100:.1f}, "
                       f"Isolation Forest {p_if * 100:.1f} benign percentile)")
    msg = f"{head} The flow is {' and '.join(drivers)}."
    if evidence:
        msg += " Main deviations: " + _phrases(evidence) + "."
    return msg


def _phrases(evidence: list[dict]) -> str:
    seen, parts = set(), []
    for ev in evidence:
        cat = feature_category(ev["feature"])
        if ev["severity"] == "info":
            phrase = f"atypical {cat} for comparable benign flows ({ev['feature']})"
        else:
            phrase = f"unusually {'high' if ev['direction'] == 'above' else 'low'} {cat} ({ev['feature']})"
        if (cat, ev["direction"], ev["severity"] == "info") in seen:
            continue
        seen.add((cat, ev["direction"], ev["severity"] == "info"))
        parts.append(phrase)
    return "; ".join(parts[:4])
