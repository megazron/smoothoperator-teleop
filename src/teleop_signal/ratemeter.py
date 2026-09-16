"""Measure the rate a stream ACTUALLY arrives at. Never assume the configured one.

The case that motivated this: a RealSense over a usbip link delivered
~5 Hz with gaps up to 0.93 s. The GUI's STALE threshold was 0.5 s, so the
scene-camera panel flickered STALE on those gaps while the camera node itself
reported no dropout -- because the node counted frames, not gaps. The rig's
Kortex relay had a configured `rate_hz` of 30 and achieved 18.4-18.7 because
each cycle was two sequential network round trips. Both numbers were only
knowable by measuring, so: feed every timestamp here and read the answer.

Feed MONOTONIC timestamps (time.monotonic()). A wall clock steps backwards on
host resync -- one WSL run measured a send latency of -2321 ms that way.
"""
from __future__ import annotations

from collections import deque


class RateMeter:
    def __init__(self, stale_after_s: float = 0.5, window_s: float = 5.0):
        self.stale_after = float(stale_after_s)
        self.window = float(window_s)
        self._t = deque()
        self.count = 0
        self.max_gap = 0.0
        self.stale_gaps = 0
        self.last = None
        self._t0 = None

    def reset(self):
        self.__init__(self.stale_after, self.window)

    def tick(self, t: float):
        t = float(t)
        if self.last is not None:
            gap = t - self.last
            if gap < 0:
                raise ValueError(
                    "timestamp went backwards by %.3f s -- feed monotonic "
                    "time, not the wall clock" % -gap)
            if gap > self.max_gap:
                self.max_gap = gap
            if gap > self.stale_after:
                self.stale_gaps += 1
        else:
            self._t0 = t
        self.last = t
        self.count += 1
        self._t.append(t)
        while self._t and t - self._t[0] > self.window:
            self._t.popleft()

    def hz(self) -> float:
        """Achieved rate over the sliding window."""
        if len(self._t) < 2:
            return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / span if span > 0 else 0.0

    def mean_hz(self) -> float:
        """Achieved rate over the whole stream."""
        if self.count < 2 or self.last is None or self._t0 is None:
            return 0.0
        span = self.last - self._t0
        return (self.count - 1) / span if span > 0 else 0.0

    def is_stale(self, now: float) -> bool:
        return self.last is None or (float(now) - self.last) > self.stale_after

    def summary(self) -> dict:
        return {"count": self.count, "hz_window": round(self.hz(), 2),
                "hz_mean": round(self.mean_hz(), 2),
                "max_gap_s": round(self.max_gap, 3),
                "gaps_over_stale": self.stale_gaps,
                "stale_after_s": self.stale_after}

    def format(self) -> str:
        s = self.summary()
        line = ("%d samples  %.2f Hz mean  %.2f Hz last %.0fs  max gap %.3f s  "
                "%d gap(s) > %.2f s stale threshold"
                % (s["count"], s["hz_mean"], s["hz_window"], self.window,
                   s["max_gap_s"], s["gaps_over_stale"], s["stale_after_s"]))
        if s["gaps_over_stale"]:
            line += ("\n  a STALE indicator with this threshold flickers on this "
                     "stream even though every frame arrives; raise the "
                     "threshold above %.2f s or fix the link" % s["max_gap_s"])
        return line
