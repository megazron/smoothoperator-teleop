import math

import numpy as np

from teleop_signal import alignment
from teleop_signal.channels import circular_range, verdict


def test_yaw_recovered_from_synthetic_motion():
    yaw = 37.5
    R = alignment.yaw_matrix(-yaw)                       # operator frame is robot rotated by -yaw
    towards_robot = np.array([0.0, -1.0, 0.0])
    measured = R @ towards_robot * 0.3
    got = alignment.solve_yaw_deg(measured[:2], towards_robot[:2])
    assert abs(got - yaw) < 0.1
    right = R @ np.array([1.0, 0.0, 0.0]) * 0.2
    res = alignment.assess(measured, right, got)
    assert res["refusals"] == []
    assert abs(res["angle_between_motions_deg"] - 90.0) < 0.1


def test_refusals_for_short_and_tilted_motion():
    r = alignment.assess(np.array([0.05, 0, 0]), None, 0.0)
    assert any("50 mm" in m for m in r["refusals"])
    r = alignment.assess(np.array([0.2, 0, 0.2 * math.tan(math.radians(40))]), None, 0.0)
    assert any("off horizontal" in m for m in r["refusals"])


def test_non_perpendicular_check_motion_is_refused():
    r = alignment.assess(np.array([0.3, 0, 0]), np.array([0.3, 0.05, 0]), 0.0)
    assert any("not ~90" in m for m in r["refusals"])


def test_fit_rotation_pure_yaw_and_mirror():
    rng = np.random.default_rng(5)
    P = rng.normal(0, 0.3, (30, 3))
    R = alignment.yaw_matrix(25.0)
    Q = P @ R.T + np.array([0.1, 0.2, 0.0])
    r = alignment.fit_rotation(P, Q)
    assert not r["is_reflection"]
    assert abs(r["yaw_deg"] - 25.0) < 1e-6
    assert r["rms_m"] < 1e-9
    M = np.diag([-1.0, 1.0, 1.0]) @ R               # "mirror them"
    r2 = alignment.fit_rotation(P, P @ M.T)
    assert r2["is_reflection"]
    assert r2["orthogonal_det"] < 0
    assert "REFLECTION" in r2["message"]
    assert r2["reflection_rms_m"] < r2["rms_m"]


def _alive(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = 90 + 60 * np.sin(np.linspace(0, 6, n))
    return np.repeat(v[::4], 4)[:n] + rng.normal(0, 0.3, n)


def test_channel_verdicts():
    assert verdict(_alive())[0] == "ALIVE"
    assert verdict(np.full(2000, 181.0))[0] == "DEAD"
    rng = np.random.default_rng(1)
    inc = np.repeat(rng.uniform(5, 355, 500), 4)
    assert verdict(inc)[0] == "INCOHERENT"
    v = _alive(seed=2)
    v[rng.random(2000) < 0.05] = 0.0
    assert verdict(v)[0] == "INTERMITTENT"
    assert verdict(np.zeros(2000))[0] == "DEAD"          # 100% dropouts
    small = 100 + 5 * np.sin(np.linspace(0, 6, 2000))
    assert verdict(small)[0] == "SUSPECT"


def test_circular_range_across_seam():
    v = np.concatenate([np.linspace(350, 360, 50) % 360, np.linspace(0, 10, 50)])
    assert abs(circular_range(v) - 20.0) < 1e-6
    assert circular_range(np.array([10.0, 200.0])) == 170.0
