import numpy as np


METERS_PER_TICK = 0.0022


def encoder_to_v(encoder_counts, dt, meters_per_tick=METERS_PER_TICK):
    """
    Convert 4-wheel encoder counts to linear velocity.
    encoder_counts: shape (4,) in order [FR, FL, RR, RL]
    dt: time step in seconds
    """
    if dt <= 0:
        return 0.0
    fr, fl, rr, rl = encoder_counts
    dr = (fr + rr) / 2.0 * meters_per_tick
    dl = (fl + rl) / 2.0 * meters_per_tick
    return (dr + dl) / (2.0 * dt)


def integrate_step(pose, v, w, dt):
    """
    One-step SE(2) integration with body-frame linear velocity v and yaw rate w.
    pose: [x, y, theta]
    """
    x, y, theta = pose
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += w * dt
    return np.array([x, y, theta])
