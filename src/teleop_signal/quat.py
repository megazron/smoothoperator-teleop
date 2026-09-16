"""Rotation smoothing that is not a component-wise EMA.

Averaging quaternion components and renormalising is only approximately a
rotation average and fails outright across a sign flip: q and -q are the same
rotation, and a tracking runtime is free to hand you either at any moment. A
component filter that sees the sign jump interpolates the long way round and
swings the wrist through 360 deg for no physical reason. It is the nastiest
bug in quaternion smoothing because the input was perfectly valid.

Everything here canonicalises the sign against the previous OUTPUT first, then
slerps. Quaternions are (x, y, z, w).
"""
from __future__ import annotations

import math

import numpy as np

from .one_euro import LowPass, alpha_for

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def q_norm(q):
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else IDENTITY.copy()


def q_canon(q, ref):
    """`q` with the sign that puts it on the same hemisphere as `ref`."""
    q = np.asarray(q, dtype=float)
    return -q if float(np.dot(q, np.asarray(ref, dtype=float))) < 0.0 else q


def q_slerp(a, b, t: float):
    """Shortest-arc slerp; lerp when the two are nearly equal (dot > 0.9995)."""
    a = q_norm(a)
    b = q_canon(q_norm(b), a)
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if d > 0.9995:
        return q_norm(a + t * (b - a))
    th = math.acos(d)
    s = math.sin(th)
    return (math.sin((1.0 - t) * th) / s) * a + (math.sin(t * th) / s) * b


def q_angle(a, b) -> float:
    """Angle between two rotations, radians, in [0, pi]."""
    d = abs(float(np.dot(q_norm(a), q_norm(b))))
    return 2.0 * math.acos(float(np.clip(d, -1.0, 1.0)))


def q_from_axis_angle(axis, angle: float):
    ax = np.asarray(axis, dtype=float)
    ax = ax / max(np.linalg.norm(ax), 1e-12)
    s = math.sin(angle / 2.0)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, math.cos(angle / 2.0)])


class OneEuroQuat:
    """1-Euro over a rotation. Blends by slerp; the speed is angular, rad/s.

    Defaults are the ones that settled on the rig (min_cutoff 1.0, beta 3.0,
    d_cutoff 1.0) -- the same arithmetic as the position filter in radians.
    """

    def __init__(self, min_cutoff_hz: float = 1.0, beta: float = 3.0,
                 d_cutoff_hz: float = 1.0):
        self.min_cutoff = float(min_cutoff_hz)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff_hz)
        self.y = None
        self._prev = None
        self._w = LowPass()
        self.speed = 0.0
        self.cutoff = float(min_cutoff_hz)

    def reset(self):
        self.y = None
        self._prev = None
        self._w.reset()
        self.speed = 0.0
        self.cutoff = self.min_cutoff

    def __call__(self, q, dt: float):
        q = q_norm(q)
        if self.y is None:
            self.y = q.copy()
            self._prev = q.copy()
            return q.copy()
        q = q_canon(q, self.y)
        w = q_angle(q, self._prev) / dt if dt > 0 else 0.0
        self._prev = q.copy()
        self.speed = float(self._w(np.array([w]), alpha_for(dt, self.d_cutoff))[0])
        self.cutoff = self.min_cutoff + self.beta * self.speed
        self.y = q_norm(q_slerp(self.y, q, alpha_for(dt, self.cutoff)))
        return self.y.copy()
