import numpy as np
import pytest

from teleop_signal.ratemeter import RateMeter
from teleop_signal.speed_clip import PreviewDelay, SpeedClip

DT = 1.0 / 72.0


def _reach(clip):
    # a 0.5 m reach in 0.25 s (2 m/s) then hold for 1 s
    clip.step(np.zeros(3), DT)
    for i in range(1, 19):
        clip.step(np.array([0.5 * i / 18, 0, 0]), DT)
    held = None
    for _ in range(72):
        held = clip.step(np.array([0.5, 0, 0]), DT)
    return held


def test_naive_clip_keeps_debt_and_gives_back_reconverges():
    naive = SpeedClip(1.2, give_back=False)
    _reach(naive)
    assert naive.debt > 0.0
    assert naive.clipped_ticks > 0
    fair = SpeedClip(1.2, give_back=True)
    y = _reach(fair)
    assert np.allclose(y, [0.5, 0, 0], atol=1e-9)
    assert fair.debt == 0.0


def test_clip_limits_step():
    c = SpeedClip(1.0)
    c.step(np.zeros(3), DT)
    y = c.step(np.array([1.0, 0, 0]), DT)
    assert abs(np.linalg.norm(y) - 1.0 * DT) < 1e-12


def test_preview_delay():
    d = PreviewDelay(0.30)
    for i in range(100):
        d.push(i * 0.01, [i])
    assert d.get(0.05)[0] == 0                      # before the line fills: oldest
    assert d.get(0.99)[0] == pytest.approx(69, abs=1)


def test_ratemeter_hz_and_gaps():
    rm = RateMeter(stale_after_s=0.5, window_s=5.0)
    t = 0.0
    for i in range(50):
        t += 0.2 if i % 10 else 0.93            # ~5 Hz with 0.93 s gaps
        rm.tick(t)
    s = rm.summary()
    assert 3.0 < s["hz_mean"] < 6.0
    assert s["max_gap_s"] == pytest.approx(0.93)
    assert s["gaps_over_stale"] == 4                # i = 10, 20, 30, 40; the first tick has no gap
    assert rm.is_stale(t + 0.6) and not rm.is_stale(t + 0.1)


def test_ratemeter_rejects_wall_clock_step():
    rm = RateMeter()
    rm.tick(10.0)
    with pytest.raises(ValueError):
        rm.tick(7.679)
