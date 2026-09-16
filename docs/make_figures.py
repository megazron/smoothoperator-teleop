#!/usr/bin/env python3
"""Regenerates every figure in docs/img from the kit's own code.

    python3 docs/make_figures.py

Nothing here is drawn by hand except reflection_vs_rotation.svg. Every curve
comes out of OneEuro, FixedEma, OneEuroQuat, SpeedClip, RateMeter and the
channel verdicts, run on the example data or on constructed signals, so the
figures show what the code does rather than what it is meant to do.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(HERE, "img")
os.makedirs(OUT, exist_ok=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from teleop_signal import (OneEuro, OneEuroQuat, RateMeter,  # noqa: E402
                           SpeedClip, channel_verdict)
from teleop_signal.one_euro import FixedEma, LowPass  # noqa: E402
from teleop_signal.quat import q_angle, q_from_axis_angle, q_norm  # noqa: E402
from teleop_signal.tuner import load_csv, score, split_phases  # noqa: E402

# ---- style ----------------------------------------------------------------
RAW, EMA, EURO, ACC, BAD, GREY = "#9aa3ad", "#e07b39", "#1f77b4", "#2ca02c", "#d62728", "#555555"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "legend.frameon": False, "figure.facecolor": "white",
})
RATE = 72.0
DT = 1.0 / RATE


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def run(filt, t, P):
    out = np.empty_like(P)
    prev = t[0]
    for i in range(len(t)):
        dt = t[i] - prev if i else DT
        prev = t[i]
        out[i] = filt(P[i], dt)
    return out


# ---- 1. 1-Euro vs fixed EMA on the example trajectory ---------------------
def fig_one_euro_vs_ema():
    t, P = load_csv(os.path.join(ROOT, "examples", "hand_trajectory.csv"))
    still, fast = split_phases(t, P)
    F_euro = run(OneEuro(1.0, 10.0, 1.0), t, P)
    # an EMA chosen to be about as steady as the 1-Euro when still
    F_ema = run(FixedEma(0.12), t, P)
    j_raw, _ = score(t, P, P, still, fast)
    j_e, l_e = score(t, P, F_euro, still, fast)
    j_m, l_m = score(t, P, F_ema, still, fast)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (lo, hi), title in ((a, (1.0, 3.0), "still hand: tremor"),
                                (b, (3.9, 4.9), "0.5 m reach in 0.6 s: lag")):
        m = (t >= lo) & (t <= hi)
        ax.plot(t[m], P[m, 0] * 1000, color=RAW, lw=0.9, label="raw  (jitter %.2f mm)" % j_raw)
        ax.plot(t[m], F_ema[m, 0] * 1000, color=EMA, lw=1.4,
                label="fixed EMA α=0.12  (jitter %.2f mm, lag %.1f mm)" % (j_m, l_m))
        ax.plot(t[m], F_euro[m, 0] * 1000, color=EURO, lw=1.4,
                label="1-Euro 1.0 Hz / β 10 / 1.0 Hz  (jitter %.2f mm, lag %.1f mm)" % (j_e, l_e))
        ax.set_title(title)
        ax.set_xlabel("time [s]")
    a.set_ylabel("x [mm]")
    a.set_ylim(-6, 6)
    b.legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Same steadiness when still, a fraction of the lag when moving "
                 "(examples/hand_trajectory.csv, 72 Hz)", fontsize=10)
    save(fig, "one_euro_vs_ema.png")


# ---- 2. cutoff vs speed and the Nyquist line --------------------------------
def fig_cutoff_vs_speed():
    v = np.linspace(0, 1.5, 300)
    nyq = RATE / 2
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for beta, c, lab in ((0.35, BAD, "β = 0.35  (paper's pixel value, in metres: never opens)"),
                         (10.0, EURO, "β = 10  (settled)"),
                         (100.0, EMA, "β = 100  (crosses Nyquist)")):
        ax.plot(v, 1.0 + beta * v, color=c, lw=1.6, label=lab)
    ax.axhline(nyq, color=GREY, ls="--", lw=1)
    ax.text(1.49, nyq + 1.2, "Nyquist at 72 Hz = 36 Hz", ha="right", color=GREY)
    v_cross = (nyq - 1.0) / 100.0
    ax.plot([v_cross], [nyq], "o", color=EMA)
    ax.annotate("inert above %.2f m/s" % v_cross, (v_cross, nyq), xytext=(0.5, 46),
                arrowprops=dict(arrowstyle="->", color=EMA), color=EMA)
    ax.set_xlabel("hand speed [m/s]")
    ax.set_ylabel("1-Euro cutoff [Hz]  = min_cutoff + β · speed")
    ax.set_ylim(0, 60)
    ax.set_title("The cutoff must open with speed but stay under Nyquist")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "cutoff_vs_speed.png")


# ---- 3. speed clip debt ------------------------------------------------------
def fig_speed_clip_debt():
    t = np.arange(0, 4.0, DT)
    x = np.zeros_like(t)
    for i, ti in enumerate(t):        # two 0.6 m reaches at ~2.5 m/s peak, out and back
        for t0, sgn in ((0.5, +1), (2.3, -1)):
            if t0 <= ti < t0 + 0.4:
                s = (ti - t0) / 0.4
                x[i] += sgn * 0.6 * (3 * s ** 2 - 2 * s ** 3)
            elif ti >= t0 + 0.4:
                x[i] += sgn * 0.6
    P = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    runs = {}
    for key, clip in (("naive 1.20 m/s, no give-back", SpeedClip(1.20, give_back=False)),
                      ("2.00 m/s with give-back", SpeedClip(2.00, give_back=True))):
        ys, debt = [], []
        for i in range(len(t)):
            ys.append(clip.step(P[i], DT)[0])
            debt.append(clip.debt)
        runs[key] = (np.array(ys), np.array(debt))
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
    a.plot(t, x * 1000, color=RAW, lw=1.0, label="hand")
    for (key, (y, d)), c in zip(runs.items(), (BAD, EURO)):
        a.plot(t, y * 1000, color=c, lw=1.5, label=key)
        b.plot(t, d * 1000, color=c, lw=1.5)
    a.set_ylabel("x [mm]")
    a.set_title("A naive clip throws the clipped distance away; the ledger pays it back")
    a.legend(loc="center right", fontsize=8)
    b.set_ylabel("debt [mm]")
    b.set_xlabel("time [s]")
    save(fig, "speed_clip_debt.png")


# ---- 4. quaternion sign flip -------------------------------------------------
class ComponentEma:
    """The wrong way: EMA on the four components, then renormalise."""

    def __init__(self, cutoff_hz=1.0):
        self.cutoff = cutoff_hz
        self.lp = LowPass()

    def __call__(self, q, dt):
        from teleop_signal import alpha_for
        return q_norm(self.lp(np.asarray(q, float), alpha_for(dt, self.cutoff)))


def fig_quat_sign_flip():
    t = np.arange(0, 6.0, DT)
    ang = np.radians(40.0) * np.sin(2 * math.pi * 0.25 * t)      # slow wrist roll
    Q_true = np.array([q_from_axis_angle([0, 0, 1], a) for a in ang])
    Q_in = Q_true.copy()
    Q_in[t >= 3.0] *= -1.0                                         # runtime flips the sign
    ema, euro = ComponentEma(1.0), OneEuroQuat(1.0, 3.0, 1.0)
    e_ema, e_euro = [], []
    for i in range(len(t)):
        e_ema.append(math.degrees(q_angle(ema(Q_in[i], DT), Q_true[i])))
        e_euro.append(math.degrees(q_angle(euro(Q_in[i], DT), Q_true[i])))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(t, e_ema, color=BAD, lw=1.5, label="component-wise EMA (peak %.0f°)" % max(e_ema))
    ax.plot(t, e_euro, color=EURO, lw=1.5, label="OneEuroQuat, sign canonicalised (peak %.1f°)" % max(e_euro))
    ax.axvline(3.0, color=GREY, ls="--", lw=1)
    ax.text(3.05, max(e_ema) * 0.9, "input quaternion sign flips\n(q → −q, same rotation)", color=GREY, fontsize=8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("output error vs true rotation [deg]")
    ax.set_title("q and −q are the same rotation; only one filter knows it")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "quat_sign_flip.png")


# ---- 6. channel verdict strips ----------------------------------------------
def fig_channel_verdicts():
    import csv
    path = os.path.join(ROOT, "examples", "channels.csv")
    with open(path) as fh:
        r = csv.reader(fh)
        hdr = next(r)
        rows = np.array([[float(x) for x in row] for row in r])
    col = {k: rows[:, i] for i, k in enumerate(hdr)}
    tt = col["t"]
    picks = [("l_j1", "ALIVE"), ("l_j3", "DEAD"), ("l_j6", "INTERMITTENT"), ("r_j5", "INCOHERENT")]
    colors = {"ALIVE": ACC, "DEAD": GREY, "INTERMITTENT": EMA, "INCOHERENT": BAD}
    fig, axes = plt.subplots(4, 1, figsize=(8, 6.2), sharex=True)
    for ax, (name, expect) in zip(axes, picks):
        v = col[name]
        label, why, st = channel_verdict(v)
        m = tt <= 6.0
        ax.plot(tt[m], v[m], ".", ms=2, color=colors[label], rasterized=True)
        drop = v[m] == 0.0
        if drop.any():
            ax.plot(tt[m][drop], v[m][drop], "x", ms=4, color=BAD, label="dropout marker 0.0")
            ax.legend(loc="upper right", fontsize=7)
        ax.set_ylim(-10, 370)
        ax.set_ylabel("%s [deg]" % name)
        ax.set_title("%s  →  %s   (circular range %.1f°, dropouts %.1f %%, jumps > 60° %.0f %%)"
                     % (name, label, st["range_deg"], 100 * st["dropout"], 100 * st["jump_fraction"]),
                     loc="left", color=colors[label], fontsize=9)
        assert label == expect, (name, label)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Four channels, four verdicts. Range alone would call the bottom one healthy.", fontsize=10)
    fig.tight_layout()
    save(fig, "channel_verdicts.png")


# ---- 7. rate meter and the stale threshold ----------------------------------
def fig_ratemeter():
    rng = np.random.default_rng(1)
    ts = [0.0]
    while ts[-1] < 12.0:
        gap = 0.16 if rng.random() > 0.12 else 0.93          # ~5 Hz with the measured 0.93 s gaps
        ts.append(ts[-1] + gap + rng.normal(0, 0.01))
    ts = np.array(ts)
    grid = np.arange(0, 12.0, 0.01)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for k, (thr, c) in enumerate(((0.5, BAD), (1.2, EURO))):
        m = RateMeter(stale_after_s=thr)
        stale = np.zeros_like(grid, dtype=bool)
        j = 0
        for i, g in enumerate(grid):
            while j < len(ts) and ts[j] <= g:
                m.tick(ts[j]); j += 1
            stale[i] = m.is_stale(g)
        s = m.summary()
        y = 1.0 - k * 0.45
        ax.fill_between(grid, y - 0.18, y + 0.18, where=stale, color=c, alpha=0.75, step="mid",
                        label="stale_after %.1f s  →  %d gap(s) flagged, achieved %.1f Hz, max gap %.2f s"
                        % (thr, s["gaps_over_stale"], s["hz_mean"], s["max_gap_s"]))
        ax.text(-0.15, y, "STALE\n@%.1f s" % thr, ha="right", va="center", fontsize=8, color=c)
    ax.vlines(ts, 1.35, 1.5, color=GREY, lw=0.8)
    ax.text(-0.15, 1.42, "frames", ha="right", va="center", fontsize=8, color=GREY)
    ax.set_yticks([])
    ax.set_ylim(0.2, 1.6)
    ax.set_xlim(-1.2, 12)
    ax.set_xlabel("time [s]")
    ax.set_title("A ~5 Hz stream with 0.93 s gaps: a 0.5 s threshold flickers, 1.2 s does not")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(False)
    save(fig, "ratemeter.png")


if __name__ == "__main__":
    fig_one_euro_vs_ema()
    fig_cutoff_vs_speed()
    fig_speed_clip_debt()
    fig_quat_sign_flip()
    fig_channel_verdicts()
    fig_ratemeter()
