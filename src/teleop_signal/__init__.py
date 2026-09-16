"""teleop-signal-kit: the signal layer between a human hand and a robot arm.

Nothing in this package imports ROS. The maths is tested against constructed
signals whose answer is known, and wired into whatever middleware you run.
"""
from .one_euro import OneEuro, alpha_for, nyquist_check, units_check
from .quat import OneEuroQuat, q_angle, q_canon, q_norm, q_slerp
from .speed_clip import PreviewDelay, SpeedClip
from .ratemeter import RateMeter
from .alignment import assess, fit_rotation, solve_yaw_deg, yaw_matrix
from .channels import report as channel_report
from .channels import verdict as channel_verdict
from .tuner import tune

__version__ = "0.1.0"
__all__ = [
    "OneEuro", "alpha_for", "nyquist_check", "units_check",
    "OneEuroQuat", "q_angle", "q_canon", "q_norm", "q_slerp",
    "SpeedClip", "PreviewDelay", "RateMeter",
    "assess", "fit_rotation", "solve_yaw_deg", "yaw_matrix",
    "channel_report", "channel_verdict", "tune",
]
