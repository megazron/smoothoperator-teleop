"""Speed limiting that does not throw the hand's motion away.

THE TRAP. A rate limit on the tracked point -- "never move more than
max_speed * dt per tick" -- is the obvious safety clamp. Applied naively it
DISCARDS the clipped distance: every reach faster than the limit leaves the
tracked point a little further behind the hand, and nothing gives that back.
On the rig this came from, a 1.20 m/s clip on ordinary reaches produced a lag
that grew over a run. Two fixes shipped together: a limit high enough for a
human reach (2.00 m/s) and not throwing the distance away.

`SpeedClip` tracks the clipped distance as a DEBT. With `give_back=True` it
pays the debt off on subsequent ticks at up to `max_speed`, so the tracked
point re-converges on the hand instead of drifting behind it. With
`give_back=False` it behaves like the naive clamp, and `.debt` tells you how
far behind you are.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class SpeedClip:
    """Per-tick step limiter with a clipped-distance ledger.

    Call `step(target, dt)` with the point the hand is at; it returns the
    point the arm should be commanded to this tick.
    """

    def __init__(self, max_speed: float, give_back: bool = True):
        if max_speed <= 0:
            raise ValueError("max_speed must be positive")
        self.max_speed = float(max_speed)
        self.give_back = bool(give_back)
        self.y = None
        self.debt = 0.0          # metres left behind by clipping, cumulative
        self.clipped_ticks = 0

    def reset(self, at=None):
        self.y = None if at is None else np.asarray(at, dtype=float).copy()
        self.debt = 0.0
        self.clipped_ticks = 0

    def step(self, target, dt: float):
        target = np.asarray(target, dtype=float)
        if self.y is None:
            self.y = target.copy()
            return self.y.copy()
        if dt <= 0.0:
            return self.y.copy()
        allowed = self.max_speed * dt
        delta = target - self.y
        dist = float(np.linalg.norm(delta))
        if dist <= allowed:
            # Under the limit. With give_back the remaining error IS the debt
            # being paid; without it the ledger only ever grows.
            self.y = target.copy()
            if self.give_back:
                self.debt = 0.0
            return self.y.copy()
        self.clipped_ticks += 1
        self.y = self.y + delta * (allowed / dist)
        if self.give_back:
            # Debt is the current distance to the hand: it is repaid as the
            # tracked point closes on the target at max_speed.
            self.debt = dist - allowed
        else:
            self.debt += dist - allowed
        return self.y.copy()

    __call__ = step


class PreviewDelay:
    """A fixed delay line, e.g. a 0.30 s preview so a motion generator can see
    where the hand is going. Holds (t, value) pairs; `push(t, x)` then
    `get(t)` returns the value from `seconds` ago (the oldest sample once the
    line has filled, the first sample before that)."""

    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self._buf = deque()

    def reset(self):
        self._buf.clear()

    def push(self, t: float, x):
        self._buf.append((float(t), np.asarray(x, dtype=float).copy()))

    def get(self, t: float):
        if not self._buf:
            return None
        cutoff = float(t) - self.seconds
        out = self._buf[0][1]
        while len(self._buf) > 1 and self._buf[1][0] <= cutoff:
            self._buf.popleft()
            out = self._buf[0][1]
        if self._buf[0][0] <= cutoff:
            out = self._buf[0][1]
        return out.copy()

    def __len__(self):
        return len(self._buf)
