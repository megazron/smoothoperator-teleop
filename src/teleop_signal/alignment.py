"""Relate the operator's frame to the robot's, and refuse a reflection.

THE TRAP. Controller poses arrive in a tracking frame whose heading is fixed
by wherever the headset pointed when the session started. When the operator
sits ACROSS THE ROOM FACING the robot, the intuitive fix -- "they're facing
me, so mirror it" -- is a REFLECTION: a linear map with determinant -1. It
puts positions where you expect and mirrors every ORIENTATION, so the gripper
rolls the wrong way while the hand looks right. No yaw angle can produce it,
and no yaw angle can undo it.

THE ANSWER IS ONE MEASURED YAW. Ask the operator to move their hand along one
direction that is unambiguous in the room (straight towards the robot). That
direction is known in the robot's world frame, and one known direction fixes
one unknown angle. A second motion (their own right) is recorded as a CHECK,
not an input: after the solved yaw is applied the two motions must still be
roughly perpendicular and horizontal, or the operator's frame is not a pure
yaw relative to the robot's (tilted headset, slope, wrong motion).

`fit_rotation` (Kabsch) is provided for the general case where you have paired
points in both frames. It reports det(R) and refuses to hand back a
reflection as if it were a rotation.
"""
from __future__ import annotations

import math

import numpy as np


def yaw_matrix(yaw_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def solve_yaw_deg(measured_xy, target_xy):
    """Signed yaw about +z (degrees) rotating `measured` onto `target`.

    Both are horizontal direction vectors. Returns None if either is
    degenerate."""
    m = np.asarray(measured_xy, dtype=float)[:2]
    t = np.asarray(target_xy, dtype=float)[:2]
    if np.linalg.norm(m) < 1e-9 or np.linalg.norm(t) < 1e-9:
        return None
    a = math.atan2(m[1], m[0])
    b = math.atan2(t[1], t[0])
    d = math.degrees(b - a)
    return (d + 180.0) % 360.0 - 180.0


def excursion(points):
    """Largest displacement from the FIRST point, not last-minus-first.

    The operator brings their hand back, and end-to-end would then read zero
    and silently calibrate against noise."""
    P = np.asarray(points, dtype=float)
    if len(P) < 2:
        return None
    d = P - P[0]
    i = int(np.argmax(np.linalg.norm(d, axis=1)))
    return d[i]


def assess(d_towards, d_right, yaw_deg, min_travel_m=0.10, max_tilt_deg=35.0,
           perp_tol_deg=20.0) -> dict:
    """Everything that says whether the solved yaw can be trusted.

    Returns a dict of measurements plus `refusals` (a list of plain-words
    reasons; empty means accept)."""
    d_towards = np.asarray(d_towards, dtype=float)
    horiz = float(np.linalg.norm(d_towards[:2]))
    out = {
        "travel_m": float(np.linalg.norm(d_towards)),
        "horizontal_travel_m": horiz,
        "vertical_component_m": float(d_towards[2]) if d_towards.size > 2 else 0.0,
    }
    out["tilt_deg"] = math.degrees(math.atan2(abs(out["vertical_component_m"]),
                                              max(horiz, 1e-9)))
    refusals = []
    if horiz < min_travel_m:
        refusals.append("the hand moved %.0f mm horizontally, under the %.0f mm "
                        "needed to fix a heading; move further"
                        % (horiz * 1000, min_travel_m * 1000))
    if out["tilt_deg"] > max_tilt_deg:
        refusals.append("the motion was %.0f deg off horizontal; a mostly "
                        "vertical motion carries almost no heading information"
                        % out["tilt_deg"])
    if d_right is not None and yaw_deg is not None:
        d_right = np.asarray(d_right, dtype=float)
        R = yaw_matrix(yaw_deg)
        a, b = R @ d_towards, R @ d_right
        na, nb = np.linalg.norm(a[:2]), np.linalg.norm(b[:2])
        if na > 1e-6 and nb > 1e-6:
            cosang = float(np.dot(a[:2], b[:2]) / (na * nb))
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
            out["angle_between_motions_deg"] = ang
            if abs(ang - 90.0) > perp_tol_deg:
                refusals.append(
                    "after alignment the two motions are %.0f deg apart, not "
                    "~90: the operator frame is not a pure yaw relative to the "
                    "robot (tilted headset, slope, or the wrong motion)" % ang)
        out["right_travel_m"] = float(np.linalg.norm(d_right))
    out["refusals"] = refusals
    return out


def fit_rotation(P_operator, P_robot) -> dict:
    """Kabsch fit of the map operator -> robot from paired points.

    Returns dict(R, t, det, rms_m, is_reflection, message). The best-fitting
    ORTHOGONAL map is reported with its determinant; if it is a reflection
    (det -1) the proper rotation is ALSO fitted (Kabsch sign correction) and
    both residuals are given, so you can see that the reflection fits your
    data better -- which is exactly the evidence that the operator is facing
    the robot and the setup, not the maths, needs to change.
    """
    P = np.asarray(P_operator, dtype=float)
    Q = np.asarray(P_robot, dtype=float)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[0] < 3:
        raise ValueError("need >= 3 paired 3-D points of equal count")
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    O = Vt.T @ U.T                       # best orthogonal map, may reflect
    det_o = float(np.linalg.det(O))
    D = np.diag([1.0, 1.0, math.copysign(1.0, det_o)])
    R = Vt.T @ D @ U.T                   # proper rotation
    t = qc - R @ pc

    def rms(M):
        return float(np.sqrt((((P @ M.T) + (qc - M @ pc) - Q) ** 2).sum(1).mean()))

    out = {"R": R, "t": t, "det": float(np.linalg.det(R)),
           "rms_m": rms(R), "orthogonal_det": det_o,
           "is_reflection": det_o < 0}
    if det_o < 0:
        out["reflection_rms_m"] = rms(O)
        out["message"] = (
            "the best orthogonal fit is a REFLECTION (det %.2f; reflection "
            "fits to %.1f mm, best proper rotation to %.1f mm). Facing the "
            "robot and copying it mirrors every orientation while positions "
            "look right. No yaw can express this; the fix is a measured yaw "
            "from one known direction, and re-checking which way the operator "
            "is facing." % (det_o, out["reflection_rms_m"] * 1000,
                            out["rms_m"] * 1000))
    else:
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        out["yaw_deg"] = yaw
        out["message"] = ("proper rotation (det +1), yaw %.2f deg, residual "
                          "%.1f mm" % (yaw, out["rms_m"] * 1000))
    return out
