"""A ROS-free teleop loop: hand pose in, arm command out.

Drop your middleware in where the comments say. Everything here is time-
correct, so it behaves the same at 72 Hz, 100 Hz and under load.
"""
import time
import numpy as np
from teleop_signal import OneEuro, OneEuroQuat, SpeedClip, RateMeter, nyquist_check

RATE = 72.0
pos_f = OneEuro(min_cutoff_hz=1.0, beta=10.0, d_cutoff_hz=1.0)     # metres
rot_f = OneEuroQuat(min_cutoff_hz=1.0, beta=3.0, d_cutoff_hz=1.0)  # radians
clip = SpeedClip(max_speed=2.0, give_back=True)                     # m/s, a human reach
meter = RateMeter(stale_after_s=0.5)
nyquist_check(pos_f.beta, typical_speed=1.0, sample_rate_hz=RATE)   # warns if inert

def read_hand():        # <- your tracker: returns (p_xyz [m], q_xyzw)
    t = time.monotonic()
    return np.array([0.3 * np.sin(t), 0.0, 1.0]), np.array([0, 0, np.sin(t / 2), np.cos(t / 2)])

def command_arm(p, q):  # <- your robot: publish a pose target
    pass

last = time.monotonic()
while True:
    now = time.monotonic(); dt = now - last; last = now
    meter.tick(now)
    p_raw, q_raw = read_hand()
    p = clip.step(pos_f(p_raw, dt), dt)
    q = rot_f(q_raw, dt)
    command_arm(p, q)
    if meter.count % int(RATE) == 0:
        print("%.1f Hz  cutoff %.1f Hz  speed %.2f m/s  clip debt %.0f mm"
              % (meter.hz(), pos_f.cutoff, pos_f.speed, clip.debt * 1000))
    time.sleep(max(0.0, 1.0 / RATE - (time.monotonic() - now)))
