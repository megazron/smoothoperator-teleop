import math

import numpy as np

from teleop_signal.quat import OneEuroQuat, q_angle, q_canon, q_from_axis_angle, q_slerp

DT = 1.0 / 72.0


def test_canon_flips_to_same_hemisphere():
    q = np.array([0, 0, 0.1, 0.995])
    assert np.allclose(q_canon(-q, q), q)
    assert np.allclose(q_canon(q, q), q)


def test_slerp_shortest_arc_and_endpoints():
    a = q_from_axis_angle([0, 0, 1], 0.0)
    b = q_from_axis_angle([0, 0, 1], math.radians(90))
    assert np.allclose(q_slerp(a, b, 0.0), a)
    assert abs(q_angle(q_slerp(a, b, 1.0), b)) < 1e-9
    mid = q_slerp(a, -b, 0.5)                       # -b is the same rotation
    assert abs(math.degrees(q_angle(a, mid)) - 45.0) < 1e-6


def test_sign_flip_produces_no_excursion():
    """Input flips sign mid-stream; the output must not swing through 360."""
    f = OneEuroQuat(1.0, 3.0, 1.0)
    q = q_from_axis_angle([0, 1, 0], math.radians(20))
    prev = f(q, DT)
    worst = 0.0
    for i in range(200):
        qi = q if i < 100 else -q
        out = f(qi, DT)
        worst = max(worst, math.degrees(q_angle(out, prev)))
        prev = out
    assert worst < 0.01


def test_quat_filter_converges_and_first_passes_through():
    f = OneEuroQuat()
    q0 = q_from_axis_angle([1, 0, 0], 0.3)
    assert np.allclose(f(q0, DT), q0)
    q1 = q_from_axis_angle([1, 0, 0], 1.3)
    out = None
    for _ in range(400):
        out = f(q1, DT)
    assert math.degrees(q_angle(out, q1)) < 0.5
