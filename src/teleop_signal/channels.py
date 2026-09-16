"""Health verdicts for rotary sensor channels (potentiometers, encoders).

A failing channel does not go quiet. It returns numbers that span a wide
range but are not a trajectory: consecutive updates unrelated, jumping tens
of degrees in 60 ms. Range alone calls that healthy, and that is how a broken
channel gets trusted. On the master arm this came from, 7 of 14 pot channels
were INCOHERENT from a soldering fault, and they had passed every check that
only looked at range.

THE RULES, in order, and why each exists.

  Samples are cleaned first: exact 0.0 is treated as the firmware's dropout
  marker, and anything outside [0, 360] as a parse glitch.

  RANGE is CIRCULAR: the smallest arc containing every sample, bounded by
  360 by construction. Unwrap-and-subtract is wrong for a rotary sensor:
  across a dropout gap the unwrap cannot know how far the joint travelled
  and the offset runs away. That rule once reported 314363 deg.

  STEPS are computed between DISTINCT sensor updates that are adjacent in
  time. A recorder samples faster than a pot updates, so most consecutive
  rows are bit-identical; including them drives the noise floor to exactly
  0.000 on every channel, which is a property of the recorder.

  VERDICT:
    DEAD          circular range < 5 deg, or dropouts > 90 %
    INCOHERENT    more than 5 % of updates jump > 60 deg
    INTERMITTENT  dropouts > 2 %, otherwise coherent
    ALIVE         coherent, range >= 20 deg, dropouts <= 2 %
    SUSPECT       anything else, named rather than bucketed silently
"""
from __future__ import annotations

import json

import numpy as np

JUMP_DEG = 60.0
DEAD_RANGE_DEG = 5.0
ALIVE_RANGE_DEG = 20.0
INCOHERENT_FRACTION = 0.05
DEAD_DROPOUT = 0.90
INTERMITTENT_DROPOUT = 0.02


def clean(samples):
    """(valid_values, dropout_fraction). 0.0 and out-of-range are dropouts."""
    v = np.asarray(samples, dtype=float).ravel()
    if v.size == 0:
        return v, 1.0
    good = np.isfinite(v) & (v != 0.0) & (v >= 0.0) & (v <= 360.0)
    return v[good], float(1.0 - good.mean())


def circular_range(values) -> float:
    """Smallest arc (deg) containing every value. Never exceeds 360."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0
    if v.size == 1:
        return 0.0
    a = np.sort(np.mod(v, 360.0))
    gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    return float(360.0 - gaps.max())


def update_steps(values):
    """Angular steps (deg, shortest way) between DISTINCT adjacent updates."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return np.zeros(0)
    changed = np.concatenate([[True], np.diff(v) != 0.0])
    u = v[changed]
    if u.size < 2:
        return np.zeros(0)
    dv = np.abs(np.diff(u)) % 360.0
    return np.minimum(dv, 360.0 - dv)


def verdict(samples):
    """-> (label, why, stats)."""
    vals, drop = clean(samples)
    rng = circular_range(vals)
    st = update_steps(vals)
    jump_frac = float((st > JUMP_DEG).mean()) if st.size else 0.0
    stats = {"n": int(np.asarray(samples).size), "updates": int(st.size) + 1 if st.size else int(vals.size > 0),
             "range_deg": round(rng, 2), "dropout": round(drop, 4),
             "jump_fraction": round(jump_frac, 4),
             "step_p50_deg": round(float(np.median(st)), 3) if st.size else 0.0}
    if rng < DEAD_RANGE_DEG or vals.size == 0:
        return "DEAD", "range %.1f deg, %d updates" % (rng, stats["updates"]), stats
    if drop > DEAD_DROPOUT:
        return "DEAD", "%.0f%% dropouts" % (100 * drop), stats
    if jump_frac > INCOHERENT_FRACTION:
        return ("INCOHERENT", "%.0f%% of updates jump > %.0f deg; values span "
                "%.0f deg but consecutive samples are unrelated"
                % (100 * jump_frac, JUMP_DEG, rng), stats)
    if drop > INTERMITTENT_DROPOUT:
        return "INTERMITTENT", "%.1f%% dropouts, coherent" % (100 * drop), stats
    if rng >= ALIVE_RANGE_DEG:
        return "ALIVE", "range %.0f deg" % rng, stats
    return ("SUSPECT", "coherent but only %.1f deg of range -- was the joint "
            "swept?" % rng, stats)


def report(channels: dict, baseline: dict | None = None) -> dict:
    """{name: (label, why, stats)} plus a diff against `baseline` if given.

    `baseline` is the dict returned by a previous call (or loaded from the
    JSON `save_baseline` writes)."""
    out = {}
    for name, samples in channels.items():
        label, why, stats = verdict(samples)
        entry = {"verdict": label, "why": why, **stats}
        if baseline and name in baseline:
            prev = baseline[name].get("verdict")
            entry["was"] = prev
            entry["changed"] = prev != label
        out[name] = entry
    return out


def save_baseline(rep: dict, path: str):
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)


def load_baseline(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def format_report(rep: dict) -> str:
    order = ["DEAD", "INCOHERENT", "INTERMITTENT", "SUSPECT", "ALIVE"]
    lines = ["%-16s %-12s %-8s %-7s %-6s %s" % ("channel", "verdict", "range", "drop%", "jump%", "note")]
    for name, e in rep.items():
        was = ("  (was %s)" % e["was"]) if e.get("changed") else ""
        lines.append("%-16s %-12s %7.1f %6.1f %6.1f  %s%s" % (
            name, e["verdict"], e["range_deg"], 100 * e["dropout"],
            100 * e["jump_fraction"], e["why"], was))
    counts = {k: sum(1 for e in rep.values() if e["verdict"] == k) for k in order}
    usable = counts["ALIVE"] + counts["INTERMITTENT"]
    lines.append("  " + "  ".join("%s %d" % (k, counts[k]) for k in order if counts[k]))
    lines.append("  usable (ALIVE or INTERMITTENT): %d of %d" % (usable, len(rep)))
    return "\n".join(lines)
