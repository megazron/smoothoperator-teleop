"""Regenerates examples/hand_trajectory.csv and examples/channels.csv."""
import csv, os
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3)
rate = 72.0
t = np.arange(0, 12.0, 1 / rate)
P = np.zeros((len(t), 3)); P[:, 2] = 1.0
# still for 4 s, a 0.5 m reach in 0.6 s, still 2 s, back in 0.8 s, still
for i, ti in enumerate(t):
    if 4.0 <= ti < 4.6:
        s = (ti - 4.0) / 0.6; P[i, 0] = 0.5 * (3 * s**2 - 2 * s**3)
    elif 4.6 <= ti < 6.6:
        P[i, 0] = 0.5
    elif 6.6 <= ti < 7.4:
        s = (ti - 6.6) / 0.8; P[i, 0] = 0.5 * (1 - (3 * s**2 - 2 * s**3))
P += rng.normal(0, 0.0015, P.shape)                 # 1.5 mm rms controller tremor
with open(os.path.join(here, "hand_trajectory.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["t", "x", "y", "z"])
    for ti, row in zip(t, P): w.writerow([f"{ti:.5f}"] + [f"{v:.6f}" for v in row])
# channels: 14 pots, one DEAD, one INCOHERENT, one INTERMITTENT
n = 3000; tt = np.arange(n) / 200.0
sweep = 90 + 60 * np.sin(2 * np.pi * 0.25 * tt)
cols = {"t": tt}
for arm in "lr":
    for j in range(1, 8):
        v = np.repeat(sweep[::4], 4)[:n] + rng.normal(0, 0.3, n)   # pot updates at 50 Hz
        cols[f"{arm}_j{j}"] = v
cols["l_j3"] = np.full(n, 181.2) + rng.normal(0, 0.05, n)          # DEAD: no range
inc = rng.uniform(5, 355, n // 4); cols["r_j5"] = np.repeat(inc, 4)[:n]   # INCOHERENT: random jumps
im = cols["l_j6"].copy(); im[rng.random(n) < 0.05] = 0.0; cols["l_j6"] = im   # INTERMITTENT: 5% dropouts
with open(os.path.join(here, "channels.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(list(cols))
    for i in range(n): w.writerow([f"{cols[k][i]:.3f}" for k in cols])
print("wrote examples")
