import math
import warnings

import numpy as np
import pytest

from teleop_signal.one_euro import FixedEma, OneEuro, alpha_for, nyquist_check, units_check

DT = 1.0 / 72.0


def test_alpha_edge_cases():
    assert alpha_for(0.0, 1.0) == 1.0
    assert alpha_for(-1.0, 1.0) == 1.0
    assert alpha_for(DT, math.inf) == 1.0
    assert alpha_for(DT, 0.0) == 0.0
    a = alpha_for(DT, 1.0)
    assert 0.0 < a < 1.0
    assert alpha_for(DT, 10.0) > a          # higher cutoff, less filtering


def test_first_sample_passes_through():
    f = OneEuro()
    x = np.array([0.3, -0.2, 1.1])
    assert np.allclose(f(x, DT), x)


def test_still_hand_attenuated():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.0015, (900, 3))
    f = OneEuro(1.0, 10.0, 1.0)
    out = np.array([f(n, DT) for n in noise])
    raw = float(np.sqrt((noise[200:] ** 2).sum(1).mean()))
    filt = float(np.sqrt((out[200:] ** 2).sum(1).mean()))
    assert filt < 0.5 * raw


def test_less_lag_than_ema_with_comparable_steadiness():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.0015, (900, 3))
    oe, ema = OneEuro(1.0, 10.0, 1.0), FixedEma(0.15)
    j_oe = np.sqrt((np.array([oe(n, DT) for n in noise])[200:] ** 2).sum(1).mean())
    j_ema = np.sqrt((np.array([ema(n, DT) for n in noise])[200:] ** 2).sum(1).mean())
    assert j_oe <= j_ema * 1.5           # comparably steady when still
    ramp = np.outer(np.arange(900) * DT * 0.4, [1, 0, 0])
    oe.reset(); ema.reset()
    lag_oe = np.linalg.norm(np.array([oe(p, DT) for p in ramp])[-1] - ramp[-1])
    lag_ema = np.linalg.norm(np.array([ema(p, DT) for p in ramp])[-1] - ramp[-1])
    assert lag_oe < lag_ema


def test_dt_correct_same_response_at_two_rates():
    """Half the rate, same physical signal, output within tolerance."""
    def run(rate):
        f = OneEuro(1.0, 0.0, 1.0)
        t = np.arange(0, 3, 1 / rate)
        return f, [f(np.array([1.0, 0, 0]) if ti > 0 else np.zeros(3), 1 / rate)[0] for ti in t][-1]
    _, a = run(72.0)
    _, b = run(36.0)
    assert abs(a - b) < 0.02


def test_speed_and_cutoff_exposed():
    f = OneEuro(1.0, 10.0, 1.0)
    f(np.zeros(3), DT)
    for i in range(1, 50):
        f(np.array([0.5 * i * DT, 0, 0]), DT)
    assert f.speed > 0.3
    assert f.cutoff > f.min_cutoff


def test_nyquist_warns_for_pixel_beta_in_metres():
    with pytest.warns(RuntimeWarning):
        r = nyquist_check(100.0, 0.5, 100.0)
    assert not r["ok"]
    assert nyquist_check(10.0, 0.5, 100.0)["ok"]


def test_units_check_flags_pixel_beta():
    with pytest.warns(RuntimeWarning):
        assert not units_check(0.35, "m")["ok"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert units_check(10.0, "m")["ok"]


def test_rejects_negative_parameters():
    with pytest.raises(ValueError):
        OneEuro(-1.0)
