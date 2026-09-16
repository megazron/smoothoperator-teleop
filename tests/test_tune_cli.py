import os
import subprocess
import sys

import numpy as np

from teleop_signal import tuner as tune_mod

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(os.path.dirname(HERE), "examples")


def test_tune_recommends_sane_values():
    t, P = tune_mod.load_csv(os.path.join(EX, "hand_trajectory.csv"))
    rec = tune_mod.tune(t, P, 72.0, 1.0, 10.0)
    assert rec["beta"] > 0
    assert 0.1 <= rec["min_cutoff_hz"] <= 4.0
    assert rec["still_jitter_mm"] < rec["raw_jitter_mm"]
    assert rec["lag_at_speed_mm"] < 30
    assert any(not ok for *_, ok in rec["table"])       # Nyquist was hit and stopped the sweep


def _cli(*args):
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(HERE), "src"))
    return subprocess.run([sys.executable, "-m", "teleop_signal.cli", *args],
                          capture_output=True, text=True, env=env)


def test_cli_tune_channels_selftest(tmp_path):
    r = _cli("tune", os.path.join(EX, "hand_trajectory.csv"), "--rate", "72")
    assert r.returncode == 0 and "recommended:" in r.stdout
    b = tmp_path / "b.json"
    r = _cli("channels", os.path.join(EX, "channels.csv"), "--save-baseline", str(b))
    assert r.returncode == 0 and "INCOHERENT" in r.stdout and "DEAD" in r.stdout and b.exists()
    r = _cli("channels", os.path.join(EX, "channels.csv"), "--baseline", str(b))
    assert r.returncode == 0 and "(was" not in r.stdout
    r = _cli("selftest")
    assert r.returncode == 0 and "PASSED" in r.stdout
    out = tmp_path / "f.csv"
    r = _cli("filter", os.path.join(EX, "hand_trajectory.csv"), str(out))
    assert r.returncode == 0 and out.exists()


def test_cli_rate_and_align(tmp_path):
    ts = tmp_path / "ts.txt"
    vals, t = [], 0.0
    for i in range(40):
        t += 0.93 if i % 10 == 5 else 0.2
        vals.append(str(t))
    ts.write_text("\n".join(vals))
    r = _cli("rate", str(ts), "--stale", "0.5")
    assert r.returncode == 0 and "Hz" in r.stdout
    tw = tmp_path / "towards.csv"
    yaw = np.radians(-30.0)
    rows = ["t,x,y,z"]
    for i in range(50):
        d = np.array([0, -0.3 * i / 49, 0])
        p = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]) @ d
        rows.append(f"{i*0.02},{p[0]},{p[1]},{1.0}")
    tw.write_text("\n".join(rows))
    r = _cli("align", str(tw))
    assert r.returncode == 0 and "ACCEPTED" in r.stdout and "solved yaw: 30.00" in r.stdout
