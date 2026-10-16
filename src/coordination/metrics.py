"""Evaluation metrics for the synthetic benchmark (design section 12). Retrospective; reads the sealed manifest only AFTER detector
outputs exist. Real-data outputs never use these recovery metrics.

Matching: a record c at day t matches planted group g (active at t: formation_day <= t <= active_end) if
J_t = |cards(c) & G_t| / |cards(c) | G_t| >= threshold, G_t = planted cards of g with >= 1 planted transaction on or before t.
Primary metric: track-level AUPRC at J >= 0.5 (tracks ranked by max-over-life score; a true positive recovers a not-yet-recovered
planted coordinated group of the evaluated regime; tracks that only re-match an already-recovered group, or match a group of another
regime, are neutral; tracks matching only decoys or nothing are false positives; recall denominator = all planted groups of the
evaluated regime, including groups that are structurally undetectable).
"""
from __future__ import annotations

import numpy as np

from src.coordination import frozen as F
from src.coordination.pipeline import METHODS, record_score


class GroupIndex:
    def __init__(self, manifest: dict):
        self.groups = {g["group_id"]: g for g in manifest["groups"]}
        self.card2groups: dict[int, list[int]] = {}
        for gid, g in self.groups.items():
            for c in g["cards_final"]:
                self.card2groups.setdefault(int(c), []).append(gid)
        self.first = {gid: {int(c): d for c, d in g["card_first_day"].items()} for gid, g in self.groups.items()}

    def best_match(self, cards, t: int):
        """Returns list of (gid, J) for every group sharing a card and active at t, sorted by J desc."""
        cand = set()
        cs = set(int(c) for c in cards)
        for c in cs:
            cand.update(self.card2groups.get(c, ()))
        out = []
        for gid in cand:
            g = self.groups[gid]
            if not (g["formation_day"] <= t <= g["active_end"]):
                continue
            Gt = {c for c, d in self.first[gid].items() if d <= t}
            if not Gt:
                continue
            inter = len(cs & Gt)
            out.append((gid, inter / (len(cs) + len(Gt) - inter)))
        out.sort(key=lambda x: -x[1])
        return out


def annotate(records: list, gi: GroupIndex) -> None:
    for r in records:
        m = gi.best_match(r["cards"], r["day"])
        r["matches"] = m                      # list of (gid, J) desc
        r["best_J"] = m[0][1] if m else 0.0


def _regime_ids(gi: GroupIndex, regime):
    ids = [gid for gid, g in gi.groups.items() if g["kind"] == "coordinated" and (regime is None or g["regime_alpha"] == regime)]
    return set(ids)


def method_records(out: dict, method: str, comps_override=None):
    kind, comps = METHODS[method]
    recs = out.get(kind, [])
    for r in recs:
        r["_score"] = record_score(r, comps)
    return recs


def track_ap(records: list, gi: GroupIndex, regime, thr: float) -> dict:
    """Track-level AP (see module docstring). `records` must be annotated."""
    pos = _regime_ids(gi, regime)
    tracks: dict[int, dict] = {}
    for r in records:
        tr = tracks.setdefault(r["track"], {"score": -np.inf, "match": {}, "decoy": False, "other": False})
        tr["score"] = max(tr["score"], r["_score"])
        for gid, J in r["matches"]:
            if J < thr:
                continue
            g = gi.groups.get(gid)
            if g is None:                          # group excluded by a bootstrap resample: neutral
                tr["other"] = True
            elif g["kind"] == "decoy":
                tr["decoy"] = True
            elif gid in pos:
                tr["match"][gid] = max(tr["match"].get(gid, 0.0), J)
            else:
                tr["other"] = True
    order = sorted(tracks.items(), key=lambda kv: (-kv[1]["score"], kv[0]))
    rec_groups: set = set()
    tp = fp = 0
    ap = 0.0
    prev_r = 0.0
    for tid, tr in order:
        if tr["match"]:
            new = [g for g in tr["match"] if g not in rec_groups]
            if new:
                rec_groups.add(max(new, key=lambda g: tr["match"][g]))
                tp += 1
                R = len(rec_groups) / max(len(pos), 1)
                ap += (R - prev_r) * tp / (tp + fp)
                prev_r = R
            continue                              # duplicate detection of a recovered group: neutral
        if tr["other"] and not tr["decoy"]:
            continue                              # matches only groups of another regime: neutral
        fp += 1
    return {"ap": float(ap), "n_groups": len(pos), "n_recovered": len(rec_groups), "n_tracks": len(tracks), "tp": tp, "fp": fp}


def alert_stream(records: list, gi: GroupIndex, regime, thr: float, budget: int, days: list[int]) -> dict:
    pos = _regime_ids(gi, regime)
    by_day: dict[int, list] = {}
    for r in records:
        by_day.setdefault(r["day"], []).append(r)
    detected: dict[int, int] = {}
    n_alert = n_false = 0
    for t in days:
        rs = sorted(by_day.get(t, []), key=lambda r: (-r["_score"], r["idx"]))[:budget]
        for r in rs:
            n_alert += 1
            hit = [(gid, J) for gid, J in r["matches"] if J >= thr and gi.groups[gid]["kind"] == "coordinated"]
            hit_pos = [h for h in hit if h[0] in pos]
            if hit_pos:
                for gid, J in hit_pos:
                    detected.setdefault(gid, t)
            elif not hit:
                n_false += 1
    lat = [detected[g] - gi.groups[g]["formation_day"] for g in detected]
    return {"budget": budget, "recall": len(detected) / max(len(pos), 1), "n_groups": len(pos), "n_detected": len(detected),
            "false_alerts_per_day": n_false / max(len(days), 1), "alerts_per_day": n_alert / max(len(days), 1),
            "mean_latency_days": float(np.mean(lat)) if lat else float("nan"), "detected_day": {int(k): int(v) for k, v in detected.items()}}


def decoy_alert_rate(records: list, gi: GroupIndex, thr: float, budget: int, days: list[int]) -> dict:
    """Alerts per day whose best match is a D1/D2 decoy group (decoy discrimination for H5b)."""
    by_day: dict[int, list] = {}
    for r in records:
        by_day.setdefault(r["day"], []).append(r)
    n = 0
    for t in days:
        for r in sorted(by_day.get(t, []), key=lambda r: (-r["_score"], r["idx"]))[:budget]:
            if any(J >= thr and gi.groups[g]["kind"] == "decoy" and gi.groups[g]["pattern"] in ("D1", "D2") for g, J in r["matches"]):
                n += 1
    return {"decoy_alerts_per_day": n / max(len(days), 1)}


def membership_quality(records: list, gi: GroupIndex, thr: float) -> dict:
    best: dict[int, tuple] = {}
    for r in records:
        for gid, J in r["matches"]:
            if J >= thr and gi.groups[gid]["kind"] == "coordinated" and (gid not in best or J > best[gid][0]):
                Gt = {c for c, d in gi.first[gid].items() if d <= r["day"]}
                cs = set(r["cards"].tolist())
                best[gid] = (J, len(cs & Gt) / len(cs), len(cs & Gt) / len(Gt))
    if not best:
        return {"n": 0}
    p = np.mean([b[1] for b in best.values()])
    rc = np.mean([b[2] for b in best.values()])
    return {"n": len(best), "precision": float(p), "recall": float(rc), "f1": float(2 * p * rc / max(p + rc, 1e-12))}


def lifecycle_metrics(records: list, tracker, gi: GroupIndex, thr: float, last_day: int) -> dict:
    gm: dict[int, set] = {}
    for r in records:
        for gid, J in r["matches"]:
            if J >= thr and gi.groups[gid]["kind"] == "coordinated":
                gm.setdefault(gid, set()).add(r["track"])
    form_err, frag, dis_delay = [], [], []
    for gid, tids in gm.items():
        g = gi.groups[gid]
        first = min(tracker.tracks[t].first_day for t in tids)
        form_err.append(first - g["formation_day"])
        frag.append(len(tids) - 1)
        for t in tids:
            tr = tracker.tracks[t]
            if tr.last_day < last_day:
                dis_delay.append(tr.last_day + F.G_DISAPPEAR - g["active_end"])
    return {"n_groups_matched": len(gm), "formation_day_error_mean": float(np.mean(form_err)) if form_err else float("nan"),
            "id_switches_per_group_mean": float(np.mean(frag)) if frag else float("nan"),
            "disappearance_declaration_delay_mean": float(np.mean(dis_delay)) if dis_delay else float("nan")}


def bootstrap_ap(records: list, gi: GroupIndex, regime, thr: float, n_boot: int, seed: int) -> tuple[float, float]:
    """95% bootstrap interval over planted groups within a seed (groups resampled; tracks keep their match sets)."""
    rng = np.random.default_rng(seed)
    pos = sorted(_regime_ids(gi, regime))
    if not pos:
        return (float("nan"), float("nan"))
    vals = []
    base = {g: gi.groups[g] for g in gi.groups}
    for _ in range(n_boot):
        pick = set(int(x) for x in rng.choice(pos, size=len(pos), replace=True))
        sub = _SubIndex(gi, pick)
        vals.append(track_ap(records, sub, regime=None, thr=thr)["ap"])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


class _SubIndex:
    """View of a GroupIndex in which only `keep` coordinated groups count as positives (decoys unchanged)."""
    def __init__(self, gi: GroupIndex, keep: set):
        self.groups = {g: v for g, v in gi.groups.items() if v["kind"] == "decoy" or g in keep}
        self.card2groups = gi.card2groups
        self.first = gi.first
