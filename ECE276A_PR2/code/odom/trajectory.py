import numpy as np

from .motion_model import encoder_to_v, integrate_step


def interp_imu_yaw(imu_w, imu_t, target_t):
    # Use IMU yaw rate (z-axis). Linear interpolation to encoder timestamps.
    yaw_rate = imu_w[2, :]
    return np.interp(target_t, imu_t, yaw_rate)


def integrate_trajectory(encoder_counts, encoder_stamps, imu_w, imu_stamps):
    """
    Integrate trajectory from encoder counts and IMU yaw rate.
    Returns poses array of shape (N, 3).
    """
    # Enforce timestamp ordering for robust async sensor alignment.
    enc_order = np.argsort(encoder_stamps)
    encoder_stamps = encoder_stamps[enc_order]
    encoder_counts = encoder_counts[:, enc_order]

    imu_order = np.argsort(imu_stamps)
    imu_stamps = imu_stamps[imu_order]
    imu_w = imu_w[:, imu_order]

    yaw_rate = interp_imu_yaw(imu_w, imu_stamps, encoder_stamps)
    n = encoder_stamps.shape[0]
    poses = np.zeros((n, 3))

    for i in range(1, n):
        dt = encoder_stamps[i] - encoder_stamps[i - 1]
        if dt <= 0:
            poses[i] = poses[i - 1]
            continue
        v = encoder_to_v(encoder_counts[:, i], dt)
        w = yaw_rate[i]
        poses[i] = integrate_step(poses[i - 1], v, w, dt)

    return poses
