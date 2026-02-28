import os
from pathlib import Path

import sys
import numpy as np

# Avoid matplotlib cache errors in restricted environments.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
RESULT_DIR = CODE_DIR / "result"
RESULT_DIR.mkdir(exist_ok=True)

from odom.trajectory import integrate_trajectory


def load_dataset(dataset):
    base = Path(__file__).resolve().parents[2] / "data"

    enc = np.load(base / f"Encoders{dataset}.npz")
    imu = np.load(base / f"Imu{dataset}.npz")

    encoder_counts = enc["counts"]  # 4 x N
    encoder_stamps = enc["time_stamps"]

    imu_angular_velocity = imu["angular_velocity"]  # 3 x M
    imu_stamps = imu["time_stamps"]

    return encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps


def interp_imu_yaw(imu_w, imu_t, target_t):
    # Use IMU yaw rate (z-axis). Linear interpolation to encoder timestamps.
    yaw_rate = imu_w[2, :]
    return np.interp(target_t, imu_t, yaw_rate)


def integrate_odom(encoder_counts, encoder_stamps, imu_w, imu_stamps):
    return integrate_trajectory(encoder_counts, encoder_stamps, imu_w, imu_stamps)


def main():
    dataset = 20
    encoder_counts, encoder_stamps, imu_w, imu_stamps = load_dataset(dataset)
    poses = integrate_odom(encoder_counts, encoder_stamps, imu_w, imu_stamps)
    print(dataset)
    plt.figure(figsize=(6, 6))
    plt.plot(poses[:, 0], poses[:, 1], "k-", linewidth=1)
    plt.axis("equal")
    plt.title(f"Odometry Trajectory (dataset {dataset})")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    out = RESULT_DIR / f"odom_traj_dataset{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved trajectory plot: {out}")

    try:
        plt.show()
    except Exception:
        # Non-GUI backend; image already saved.
        pass


if __name__ == "__main__":
    main()
