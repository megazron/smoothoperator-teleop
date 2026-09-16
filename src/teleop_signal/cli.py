"""teleop-signal command line.

  teleop-signal filter in.csv out.csv [--min-cutoff 1 --beta 10 --d-cutoff 1]
  teleop-signal tune traj.csv [--rate 72 --target-jitter-mm 1 --target-lag-mm 10]
  teleop-signal rate --stdin [--stale 0.5]        # one monotonic timestamp per line
  teleop-signal align towards.csv [right.csv] [--target-xy 0,-1]
  teleop-signal channels data.csv [--baseline b.json] [--save-baseline b.json]
  teleop-signal selftest
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from . import alignment, channels
from . import tuner as tune_mod
from .one_euro import OneEuro, nyquist_check, units_check
from .ratemeter import RateMeter


def _cmd_filter(a):
    t, P = tune_mod.load_csv(a.input)
    units_check(a.beta, "m", warn=True, min_cutoff_hz=a.min_cutoff)
    F = tune_mod.run_filter(t, P, a.min_cutoff, a.beta, a.d_cutoff)
    with open(a.output, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "x", "y", "z"])
        for ti, row in zip(t, F):
            w.writerow([f"{ti:.6f}"] + [f"{v:.6f}" for v in row])
    print("filtered %d samples -> %s" % (len(t), a.output))
    return 0


def _cmd_tune(a):
    t, P = tune_mod.load_csv(a.input)
    rec = tune_mod.tune(t, P, a.rate, a.target_jitter_mm, a.target_lag_mm, a.d_cutoff)
    print(tune_mod.format_tune(rec))
    return 0


def _cmd_rate(a):
    rm = RateMeter(a.stale, a.window)
    src = sys.stdin if a.stdin or not a.file else open(a.file)
    for line in src:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rm.tick(float(line.split(",")[0]))
    print(rm.format())
    return 0


def _load_points(path):
    with open(path) as fh:
        r = csv.reader(fh)
        rows = []
        for row in r:
            try:
                rows.append([float(x) for x in row[:4]])
            except ValueError:
                continue
    a = np.asarray(rows, dtype=float)
    return a[:, 1:4] if a.shape[1] >= 4 else a[:, :3]


def _cmd_align(a):
    tx, ty = (float(v) for v in a.target_xy.split(","))
    d_t = alignment.excursion(_load_points(a.towards))
    d_r = alignment.excursion(_load_points(a.right)) if a.right else None
    yaw = alignment.solve_yaw_deg(d_t[:2], (tx, ty))
    res = alignment.assess(d_t, d_r, yaw)
    print("towards-robot excursion: %s m" % np.round(d_t, 4))
    if d_r is not None:
        print("operator-right excursion: %s m" % np.round(d_r, 4))
    print("solved yaw: %s deg (a rotation about +z; a reflection is impossible by construction)"
          % ("%.2f" % yaw if yaw is not None else "undefined"))
    for k, v in res.items():
        if k != "refusals":
            print("  %-28s %s" % (k, round(v, 4) if isinstance(v, float) else v))
    if res["refusals"]:
        print("REFUSED:")
        for r in res["refusals"]:
            print("  - " + r)
        return 1
    print("ACCEPTED")
    return 0


def _cmd_channels(a):
    with open(a.input) as fh:
        r = csv.reader(fh)
        header = next(r)
        cols = {h: [] for h in header}
        for row in r:
            for h, v in zip(header, row):
                try:
                    cols[h].append(float(v))
                except ValueError:
                    cols[h].append(float("nan"))
    if a.time_column and a.time_column in cols:
        cols.pop(a.time_column)
    base = channels.load_baseline(a.baseline) if a.baseline else None
    rep = channels.report({k: np.asarray(v) for k, v in cols.items()}, base)
    print(channels.format_report(rep))
    if a.save_baseline:
        channels.save_baseline(rep, a.save_baseline)
        print("baseline saved -> %s" % a.save_baseline)
    return 0


def _cmd_selftest(a):
    """Constructed signals; the ground truth is arithmetic."""
    from .one_euro import FixedEma
    rng = np.random.default_rng(7)
    dt, n = 1.0 / 72.0, 900
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print("  %-60s %s %s" % (name, "PASS" if cond else "FAIL", detail))

    noise = rng.normal(0.0, 0.0015, (n, 3))
    outs = {}
    for name, f in (("one_euro", OneEuro(1.0, 10.0, 1.0)), ("ema", FixedEma(0.6))):
        outs[name] = np.array([f(noise[i], dt) for i in range(n)])
    rms = {k: float(np.sqrt((v[200:] ** 2).sum(1).mean())) * 1000 for k, v in outs.items()}
    check("still hand: 1-Euro steadier than raw 1.5 mm tremor", rms["one_euro"] < 1.0, "%.2f mm" % rms["one_euro"])
    ramp = np.outer(np.arange(n) * dt * 0.4, [1, 0, 0])
    lag = {}
    for name, f in (("one_euro", OneEuro(1.0, 10.0, 1.0)), ("ema", FixedEma(0.15))):
        F = np.array([f(ramp[i], dt) for i in range(n)])
        lag[name] = float(np.linalg.norm(F[-1] - ramp[-1])) * 1000
    check("0.4 m/s reach: 1-Euro lags less than a comparably-steady EMA",
          lag["one_euro"] < lag["ema"], "%.1f vs %.1f mm" % (lag["one_euro"], lag["ema"]))
    nq = nyquist_check(100.0, 0.5, 100.0, warn=False)
    check("beta=100 in metres at 0.5 m/s crosses 50 Hz Nyquist", not nq["ok"], "%.0f Hz" % nq["cutoff_hz"])
    print("selftest %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="teleop-signal", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    f = sp.add_parser("filter", help="run the 1-Euro over a t,x,y,z CSV")
    f.add_argument("input"); f.add_argument("output")
    f.add_argument("--min-cutoff", type=float, default=1.0)
    f.add_argument("--beta", type=float, default=10.0)
    f.add_argument("--d-cutoff", type=float, default=1.0)
    f.set_defaults(fn=_cmd_filter)

    t = sp.add_parser("tune", help="the author's procedure on a recording")
    t.add_argument("input")
    t.add_argument("--rate", type=float, default=None, help="sample rate Hz (default: from timestamps)")
    t.add_argument("--target-jitter-mm", type=float, default=1.0)
    t.add_argument("--target-lag-mm", type=float, default=10.0)
    t.add_argument("--d-cutoff", type=float, default=1.0)
    t.set_defaults(fn=_cmd_tune)

    r = sp.add_parser("rate", help="achieved rate and gaps from timestamps")
    r.add_argument("file", nargs="?")
    r.add_argument("--stdin", action="store_true")
    r.add_argument("--stale", type=float, default=0.5)
    r.add_argument("--window", type=float, default=5.0)
    r.set_defaults(fn=_cmd_rate)

    al = sp.add_parser("align", help="solve operator yaw from a towards-robot motion")
    al.add_argument("towards"); al.add_argument("right", nargs="?")
    al.add_argument("--target-xy", default="0,-1", help="towards-robot direction in the robot frame")
    al.set_defaults(fn=_cmd_align)

    c = sp.add_parser("channels", help="DEAD/INCOHERENT/INTERMITTENT/ALIVE per column")
    c.add_argument("input")
    c.add_argument("--baseline"); c.add_argument("--save-baseline")
    c.add_argument("--time-column", default="t")
    c.set_defaults(fn=_cmd_channels)

    s = sp.add_parser("selftest", help="constructed-signal checks")
    s.set_defaults(fn=_cmd_selftest)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
