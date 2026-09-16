"""dt-correct 1-Euro filter (Casiez, Roussel & Vogel, CHI 2012) over vectors.

WHY NOT A FIXED EMA. A first-order low pass with one constant alpha trades
jitter rejection directly against lag: alpha high lets a still controller's
millimetre wander reach the arm, alpha low puts a constant offset behind a
moving hand (phase lag is proportional to velocity). They are not the same
signal -- tremor is high frequency at low amplitude, motion is low frequency
at high amplitude -- so no single alpha fixes both.

THE 1-EURO ANSWER. A one-pole low pass whose cutoff rises with speed:

    cutoff = min_cutoff + beta * |estimated speed|

Slow or still, the cutoff is low and tremor is filtered hard. Moving, the
cutoff opens and the lag collapses.

dt-CORRECT. Every alpha is computed from the measured dt, so the frequency
response is a property of the filter and not of the scheduler. A constant
alpha applied once per tick filters by a different amount at 72 Hz, 100 Hz
and whatever the machine manages under load.

MIND THE UNITS OF BETA. The paper's examples are in PIXELS and quote beta in
the 0.001-1 range. A signal in METRES is ~1000x smaller for the same physical
motion, so a beta copied from the paper pins the cutoff at min_cutoff and the
filter degenerates into a very heavy fixed low pass. Measured on the rig this
came from: beta=0.35 in metres gave 55.8 mm of lag at 0.40 m/s, against
3.7 mm for the EMA it was meant to replace. See `units_check`.

MIND NYQUIST TOO. Going the other way, beta=100 in metres pushed the cutoff
past the 50 Hz Nyquist limit of a 100 Hz loop above 0.5 m/s. Past Nyquist a
one-pole filter is INERT -- alpha saturates at 1 -- so the filter did nothing
while the hand moved and only acted when it stopped. See `nyquist_check`.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

TWO_PI = 2.0 * math.pi


def alpha_for(dt: float, cutoff_hz: float) -> float:
    """One-pole coefficient giving `cutoff_hz` at a step of `dt` seconds.

    tau = 1 / (2 pi fc);  alpha = dt / (tau + dt), in (0, 1].

    alpha = 1 means "no filtering", and that is the direction this fails in:
    a non-positive dt or an infinite cutoff passes the hand straight through.
    A filter that stops filtering is annoying; one that clamps the other way
    freezes the arm.
    """
    if not dt > 0.0 or not math.isfinite(dt):
        return 1.0
    if not math.isfinite(cutoff_hz):
        return 1.0
    if cutoff_hz <= 0.0:
        return 0.0
    tau = 1.0 / (TWO_PI * cutoff_hz)
    return float(dt / (tau + dt))


class LowPass:
    """One-pole low pass over a scalar or vector, driven by an explicit alpha."""

    def __init__(self):
        self.y = None

    def reset(self):
        self.y = None

    def __call__(self, x, a: float):
        x = np.asarray(x, dtype=float)
        if self.y is None:
            self.y = x.copy()
        else:
            self.y = a * x + (1.0 - a) * self.y
        return self.y


class OneEuro:
    """1-Euro filter over a vector (a position in metres, typically).

    Parameters
    ----------
    min_cutoff_hz : cutoff when the hand is still. LOWER = steadier when
        still; the only cost is lag at low speed, where lag is cheap.
        1.0 Hz is the paper's suggested starting point.
    beta : how fast the cutoff opens with speed, in Hz per unit of speed
        (Hz per m/s for a signal in metres). HIGHER = less lag when moving,
        at the price of letting tremor through during fast motion, where
        it is invisible anyway. Tune in factors of ten.
    d_cutoff_hz : cutoff of the low pass on the speed ESTIMATE. The estimate
        is a finite difference and noisier than the signal; without this the
        cutoff would be modulated by noise. 1.0 Hz is a good default; the
        paper's 1 Hz. Too low (0.2) and the filter is late to open and late
        to settle.

    The first sample is PASSED THROUGH, never blended toward zero. Seeding at
    the origin would command the arm from (0, 0, 0) to the hand on the first
    tick after every engage.

    Attributes `speed` and `cutoff` expose the filter's current state for
    telemetry.
    """

    def __init__(self, min_cutoff_hz: float = 1.0, beta: float = 10.0,
                 d_cutoff_hz: float = 1.0):
        if min_cutoff_hz < 0 or beta < 0 or d_cutoff_hz < 0:
            raise ValueError("1-Euro parameters must be non-negative")
        self.min_cutoff = float(min_cutoff_hz)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff_hz)
        self._x = LowPass()
        self._dx = LowPass()
        self._prev = None
        self.speed = 0.0
        self.cutoff = float(min_cutoff_hz)

    def reset(self):
        self._x.reset()
        self._dx.reset()
        self._prev = None
        self.speed = 0.0
        self.cutoff = self.min_cutoff

    def __call__(self, x, dt: float):
        x = np.asarray(x, dtype=float)
        if self._prev is None:
            self._prev = x.copy()
            self._x.y = x.copy()
            return x.copy()
        if dt > 0.0:
            dx = (x - self._prev) / dt
        else:
            dx = np.zeros_like(x)
        self._prev = x.copy()
        dxh = self._dx(dx, alpha_for(dt, self.d_cutoff))
        self.speed = float(np.linalg.norm(dxh))
        self.cutoff = self.min_cutoff + self.beta * self.speed
        return self._x(x, alpha_for(dt, self.cutoff)).copy()


class FixedEma:
    """The fixed-alpha EMA the 1-Euro replaces. Kept for A/B comparison.

    Deliberately dt-blind: one constant alpha per tick, however long the tick.
    """

    def __init__(self, alpha: float = 0.6):
        self.alpha = float(alpha)
        self.y = None

    def reset(self):
        self.y = None

    def __call__(self, x, dt: float = 0.0):
        x = np.asarray(x, dtype=float)
        if self.y is None:
            self.y = x.copy()
        else:
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y.copy()


def nyquist_check(beta: float, typical_speed: float, sample_rate_hz: float,
                  min_cutoff_hz: float = 1.0, warn: bool = True) -> dict:
    """Does the cutoff stay below Nyquist at a typical working speed?

    Returns a dict with `cutoff_hz`, `nyquist_hz`, `ok`, and `speed_at_nyquist`
    (the speed above which the filter goes inert). Emits a RuntimeWarning when
    not ok, unless `warn=False`.

    A one-pole filter whose cutoff passes sample_rate/2 has alpha ~ 1: it is
    pass-through while the hand moves and only filters when it stops, which
    is the opposite of what anyone tuned it for.
    """
    nyq = float(sample_rate_hz) / 2.0
    cutoff = float(min_cutoff_hz) + float(beta) * float(typical_speed)
    ok = cutoff < nyq
    speed_at_nyq = (nyq - min_cutoff_hz) / beta if beta > 0 else math.inf
    out = {"cutoff_hz": cutoff, "nyquist_hz": nyq, "ok": ok,
           "speed_at_nyquist": speed_at_nyq}
    if not ok and warn:
        warnings.warn(
            "1-Euro cutoff %.1f Hz at %.2f units/s exceeds Nyquist %.1f Hz "
            "(sample rate %.0f Hz): the filter is INERT above %.2f units/s. "
            "Lower beta by a factor of ten." % (cutoff, typical_speed, nyq,
                                                 sample_rate_hz, speed_at_nyq),
            RuntimeWarning, stacklevel=2)
    return out


def units_check(beta: float, signal_units: str = "m",
                typical_speed: float | None = None, min_cutoff_hz: float = 1.0,
                warn: bool = True) -> dict:
    """Is beta plausible for the signal's units?

    The 1-Euro paper's worked values (beta ~ 0.001..1) are for PIXELS. In
    metres, a comparable physical motion is ~1000x smaller, so the same beta
    barely moves the cutoff. This flags a beta that cannot open the cutoff
    by even one min_cutoff at a typical speed (0.3 m/s or 30 deg/s), which
    means the filter is a heavy fixed low pass with extra steps.
    """
    default_speed = {"m": 0.3, "mm": 300.0, "px": 300.0,
                     "rad": math.radians(30.0), "deg": 30.0}
    if typical_speed is None:
        typical_speed = default_speed.get(signal_units, 1.0)
    gain = float(beta) * float(typical_speed)
    ok = gain >= float(min_cutoff_hz)
    out = {"cutoff_gain_hz_at_typical_speed": gain, "typical_speed": typical_speed,
           "ok": ok, "units": signal_units}
    if not ok and warn:
        warnings.warn(
            "beta=%.4g adds only %.3f Hz to the cutoff at %.3g %s/s; the "
            "filter cannot open and will lag like a heavy EMA. The paper's "
            "beta values are for pixels; for metres try 1-100."
            % (beta, gain, typical_speed, signal_units),
            RuntimeWarning, stacklevel=2)
    return out
