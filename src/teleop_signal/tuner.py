"""The 1-Euro author's tuning procedure, run on a recorded trajectory.

Casiez et al. tune by hand in two steps: set beta to 0 and lower min_cutoff
until a still hand gives an acceptably still output; then raise beta (in
factors of ten) until a fast motion has acceptable lag. Doing it on a
recording makes the result repeatable and writes down what each candidate
cost, and it lets Nyquist be checked at every step so a beta that makes the
filter inert is caught rather than shipped.

Input: rows of (t, x, y, z) in seconds and metres. The recording should
contain both a still period with the device's real tremor and at least one
fast reach.
"""
from __future__ import annotations

import csv
import math

import numpy as np

from .one_euro import OneEuro, nyquist_check


def load_csv(path):
    with open(path) as fh:
        r = csv.reader(fh)
        header = next(r)
        rows = [[float(x) for x in row[:4]] for row in r if row and not row[0].startswith("#")]
    a = np.asarray(rows, dtype=float)
    if a.ndim != 2 or a.shape[1] < 4:
        raise ValueError("expected columns t,x,y,z (header: %s)" % header)
    return a[:, 0], a[:, 1:4]


def run_filter(t, P, min_cutoff, beta, d_cutoff):
    f = OneEuro(min_cutoff, beta, d_cutoff)
    out = np.empty_like(P)
    prev = t[0]
    for i in range(len(t)):
        dt = t[i] - prev if i else 1.0 / 72.0
        prev = t[i]
        out[i] = f(P[i], dt)
    return out


def split_phases(t, P, still_speed=0.05, fast_speed=0.30, smooth_s=0.25):
    """Boolean masks (still, fast) from the speed of a box-smoothed position.

    Raw tremor differentiated at 72 Hz reads as ~0.15 m/s, which would call a
    still hand moving; the position is smoothed over `smooth_s` first so the
    phase decision sees the hand, not the noise."""
    t = np.asarray(t, dtype=float)
    dt = np.gradient(t)
    med = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    dt[dt <= 0] = med
    k = max(1, int(round(smooth_s / med)))
    kern = np.ones(k) / k
    Ps = np.column_stack([np.convolve(P[:, i], kern, mode="same") for i in range(P.shape[1])])
    v = np.linalg.norm(np.gradient(Ps, axis=0) / dt[:, None], axis=1)
    # edges of the box filter are biased; trim them from the still mask
    still = v < still_speed
    still[:k] = False
    still[-k:] = False
    return still, v > fast_speed


def _runs(mask):
    """Contiguous True runs of a boolean mask as (start, stop) slices."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(idx[0::2], idx[1::2]))


def score(t, P, F, still, fast, min_run=20, hp_s=0.3):
    """still-jitter: pooled RMS (mm) of the output's HIGH-FREQUENCY content
    (deviation from a `hp_s` running mean) over contiguous still segments.
    A running mean rather than a segment mean, because a heavy filter is
    still settling after a reach and that slow tail is lag, not tremor --
    lag is scored separately as the RMS distance (mm) between filtered and
    raw while fast."""
    t = np.asarray(t, dtype=float)
    med = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / 72.0
    k = max(1, int(round(hp_s / med)))
    sq, n = 0.0, 0
    for a, b in _runs(still):
        if b - a < max(min_run, 2 * k):
            continue
        seg = F[a:b]
        kern = np.ones(k) / k
        mean = np.column_stack([np.convolve(seg[:, i], kern, mode="valid") for i in range(seg.shape[1])])
        off = (k - 1) // 2
        hp = seg[off:off + len(mean)] - mean
        sq += float((hp ** 2).sum())
        n += len(hp)
    jitter = math.sqrt(sq / n) * 1000 if n else float("nan")
    lag = float("nan")
    if fast.sum() > 5:
        lag = float(np.sqrt(((F[fast] - P[fast]) ** 2).sum(1).mean())) * 1000
    return jitter, lag


def tune(t, P, rate_hz=None, target_jitter_mm=1.0, target_lag_mm=10.0,
         d_cutoff=1.0, min_cutoffs=(4.0, 2.0, 1.0, 0.5, 0.25, 0.1),
         betas=(0.0, 0.1, 1.0, 10.0, 100.0, 1000.0), verbose=False) -> dict:
    t = np.asarray(t, dtype=float)
    P = np.asarray(P, dtype=float)
    if rate_hz is None:
        rate_hz = 1.0 / float(np.median(np.diff(t)))
    still, fast = split_phases(t, P)
    raw_jit, _ = score(t, P, P, still, fast)
    table = []
    # step 1: beta = 0, walk min_cutoff DOWN until the still hand is steady
    chosen_mc = min_cutoffs[-1]
    for mc in min_cutoffs:
        j, lag = score(t, P, run_filter(t, P, mc, 0.0, d_cutoff), still, fast)
        table.append(("step1", mc, 0.0, j, lag, True))
        if not math.isnan(j) and j <= target_jitter_mm:
            chosen_mc = mc
            break
    # step 2: raise beta by x10 until the lag target is met, Nyquist checked
    typical_speed = float(np.percentile(
        np.linalg.norm(np.gradient(P, axis=0) / np.gradient(t)[:, None], axis=1)[fast], 90)) if fast.sum() > 5 else 1.0
    chosen_beta = 0.0
    for b in betas:
        nq = nyquist_check(b, typical_speed, rate_hz, chosen_mc, warn=False)
        j, lag = score(t, P, run_filter(t, P, chosen_mc, b, d_cutoff), still, fast)
        table.append(("step2", chosen_mc, b, j, lag, nq["ok"]))
        if not nq["ok"]:
            break                        # past here the filter is inert while moving
        chosen_beta = b
        if not math.isnan(lag) and lag <= target_lag_mm:
            break
    jf, lf = score(t, P, run_filter(t, P, chosen_mc, chosen_beta, d_cutoff), still, fast)
    rec = {"min_cutoff_hz": chosen_mc, "beta": chosen_beta, "d_cutoff_hz": d_cutoff,
           "still_jitter_mm": jf, "lag_at_speed_mm": lf, "raw_jitter_mm": raw_jit,
           "typical_fast_speed_mps": typical_speed, "rate_hz": rate_hz,
           "still_samples": int(still.sum()), "fast_samples": int(fast.sum()),
           "table": table}
    if verbose:
        print(format_tune(rec))
    return rec


def format_tune(rec: dict) -> str:
    lines = ["1-Euro tuning on a recorded trajectory (%.0f Hz, %d still / %d fast samples, raw still jitter %.2f mm)"
             % (rec["rate_hz"], rec["still_samples"], rec["fast_samples"], rec["raw_jitter_mm"]),
             "  %-6s %-10s %-8s %-11s %-9s %s" % ("step", "min_cutoff", "beta", "jitter_mm", "lag_mm", "nyquist")]
    for step, mc, b, j, lag, ok in rec["table"]:
        lines.append("  %-6s %-10.2f %-8g %-11.2f %-9.1f %s" % (step, mc, b, j, lag, "ok" if ok else "INERT"))
    lines.append("recommended: min_cutoff_hz=%.2f  beta=%g  d_cutoff_hz=%.1f   -> still %.2f mm, lag %.1f mm at %.2f m/s"
                 % (rec["min_cutoff_hz"], rec["beta"], rec["d_cutoff_hz"], rec["still_jitter_mm"],
                    rec["lag_at_speed_mm"], rec["typical_fast_speed_mps"]))
    if rec["beta"] == 0.0:
        lines.append("  beta stayed at 0: no beta met the lag target below Nyquist. Raise the target or the sample rate.")
    return "\n".join(lines)
