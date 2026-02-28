import os
import sys
from pathlib import Path

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

from config import LIDAR_OFFSET
from odom.trajectory import integrate_trajectory
from icp.icp_2d_scanmatch import icp_2d, transform_points_2d


def load_dataset(dataset):
    base = Path(__file__).resolve().parents[2] / "data"
    enc = np.load(base / f"Encoders{dataset}.npz")
    imu = np.load(base / f"Imu{dataset}.npz")
    lidar = np.load(base / f"Hokuyo{dataset}.npz")

    encoder_counts = enc["counts"]
    encoder_stamps = enc["time_stamps"]
    imu_w = imu["angular_velocity"]
    imu_stamps = imu["time_stamps"]

    lidar_ranges = lidar["ranges"]
    lidar_stamps = lidar["time_stamps"]
    angle_min = float(lidar["angle_min"])
    angle_inc = float(lidar["angle_increment"])
    range_min = float(lidar["range_min"])
    range_max = float(lidar["range_max"])

    return (
        encoder_counts,
        encoder_stamps,
        imu_w,
        imu_stamps,
        lidar_ranges,
        lidar_stamps,
        angle_min,
        angle_inc,
        range_min,
        range_max,
    )


def pose_to_T(pose):
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(3)
    T[:2, :2] = np.array([[c, -s], [s, c]])
    T[:2, 2] = np.array([x, y])
    return T


def T_to_pose(T):
    x, y = T[:2, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return np.array([x, y, theta])


def inv_se2(T):
    R = T[:2, :2]
    t = T[:2, 2]
    T_inv = np.eye(3)
    T_inv[:2, :2] = R.T
    T_inv[:2, 2] = -R.T @ t
    return T_inv


def nearest_pose_index(stamps, t):
    return int(np.argmin(np.abs(stamps - t)))


def ranges_to_points_lidar(ranges, angle_min, angle_inc, range_min, range_max):
    angles = angle_min + np.arange(ranges.shape[0]) * angle_inc
    valid = np.logical_and(ranges > range_min, ranges < range_max)
    r = ranges[valid]
    a = angles[valid]
    x = r * np.cos(a)
    y = r * np.sin(a)
    return np.stack([x, y], axis=1)


def lidar_to_body(points_lidar):
    # Lidar has no yaw offset in this setup, only translation in body frame.
    return points_lidar + LIDAR_OFFSET.reshape(1, 2)


def main():
    dataset = 20
    (
        encoder_counts,
        encoder_stamps,
        imu_w,
        imu_stamps,
        lidar_ranges,
        lidar_stamps,
        angle_min,
        angle_inc,
        range_min,
        range_max,
    ) = load_dataset(dataset)

    # Part 1 odometry for ICP initialization
    poses_odom = integrate_trajectory(encoder_counts, encoder_stamps, imu_w, imu_stamps)

    # Hyperparameters for runtime/robustness tradeoff
    max_iters = 25
    max_dist = 0.6
    max_pairs = lidar_ranges.shape[1] - 1

    # Initialize global pose from first lidar timestamp (nearest odom pose)
    idx0 = nearest_pose_index(encoder_stamps, lidar_stamps[0])
    T_w_curr = pose_to_T(poses_odom[idx0])
    poses_icp = [T_to_pose(T_w_curr)]
    poses_odom_lidar = [poses_odom[idx0]]
    mse_hist = []
    rel_icp = []

    # Keep first pair for before/after visualization
    first_pair = None

    for i in range(max_pairs):
        t0 = lidar_stamps[i]
        t1 = lidar_stamps[i + 1]
        idx_odom_0 = nearest_pose_index(encoder_stamps, t0)
        idx_odom_1 = nearest_pose_index(encoder_stamps, t1)
        pose_odom_0 = poses_odom[idx_odom_0]
        pose_odom_1 = poses_odom[idx_odom_1]

        # ICP init from odometry: T_{b1<-b0}
        T_w_b0 = pose_to_T(pose_odom_0)
        T_w_b1 = pose_to_T(pose_odom_1)
        T_b1_b0_init = inv_se2(T_w_b1) @ T_w_b0

        src_l = ranges_to_points_lidar(
            lidar_ranges[:, i], angle_min, angle_inc, range_min, range_max
        )
        dst_l = ranges_to_points_lidar(
            lidar_ranges[:, i + 1], angle_min, angle_inc, range_min, range_max
        )
        src_b = lidar_to_body(src_l)
        dst_b = lidar_to_body(dst_l)

        T_b1_b0_icp, hist = icp_2d(
            src_b,
            dst_b,
            init_T=T_b1_b0_init,
            max_iters=max_iters,
            max_dist=max_dist,
        )
        if len(hist) == 0:
            # fallback to odom init if ICP fails
            T_b1_b0_icp = T_b1_b0_init
            mse = np.nan
        else:
            mse = hist[-1]
        mse_hist.append(mse)
        # Relative motion for compose / pose-graph factor:
        # T_{w<-b1} = T_{w<-b0} * T_rel, where T_rel = inv(T_{b1<-b0})
        T_rel = inv_se2(T_b1_b0_icp)
        rel_icp.append(T_rel)

        if first_pair is None:
            first_pair = (src_b, dst_b, T_b1_b0_init, T_b1_b0_icp)

        # Compose global poses:
        # p1 = T_{b1<-b0} p0, and p_w = T_{w<-b} p_b
        # => T_{w<-b1} = T_{w<-b0} * inv(T_{b1<-b0})
        T_w_curr = T_w_curr @ inv_se2(T_b1_b0_icp)
        poses_icp.append(T_to_pose(T_w_curr))
        poses_odom_lidar.append(pose_odom_1)

    poses_icp = np.array(poses_icp)
    poses_odom_lidar = np.array(poses_odom_lidar)
    rel_icp = np.array(rel_icp)
    mse_hist = np.array(mse_hist)

    traj_out = RESULT_DIR / f"icp_trajectory_dataset{dataset}.npz"
    np.savez(
        traj_out,
        poses_icp=poses_icp,
        poses_odom_lidar=poses_odom_lidar,
        lidar_stamps=lidar_stamps[: poses_icp.shape[0]],
        rel_icp=rel_icp,
        icp_mse=mse_hist,
    )
    print(f"saved icp trajectory: {traj_out}")

    # 1) Before/After ICP overlay for first scan pair
    if first_pair is not None:
        src_b, dst_b, T_init, T_icp = first_pair
        src_init = transform_points_2d(src_b, T_init)
        src_after = transform_points_2d(src_b, T_icp)

        plt.figure(figsize=(6, 6))
        plt.plot(dst_b[:, 0], dst_b[:, 1], "k.", markersize=1, label="target scan t+1")
        plt.plot(src_init[:, 0], src_init[:, 1], "r.", markersize=1, label="source with odom init")
        plt.plot(src_after[:, 0], src_after[:, 1], "b.", markersize=1, label="source after ICP")
        plt.axis("equal")
        plt.legend(loc="best", fontsize=8)
        plt.title(f"Consecutive Scan Overlay: Before/After ICP (dataset {dataset})")
        out = RESULT_DIR / f"scanmatch_pair0_overlay_dataset{dataset}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved scan overlay: {out}")

    # 2) Odom vs ICP-composed trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(
        poses_odom_lidar[:, 0],
        poses_odom_lidar[:, 1],
        "r-",
        linewidth=1.0,
        label="odom (at lidar stamps)",
    )
    plt.plot(
        poses_icp[:, 0],
        poses_icp[:, 1],
        "b-",
        linewidth=1.0,
        label="ICP-composed",
    )
    plt.axis("equal")
    plt.legend(loc="best")
    plt.title(f"Trajectory: Odom vs ICP Scan Matching (dataset {dataset})")
    out = RESULT_DIR / f"scanmatch_traj_compare_dataset{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved trajectory compare: {out}")

    # 3) ICP MSE curve
    plt.figure(figsize=(7, 3))
    plt.plot(mse_hist, "k-", linewidth=1)
    plt.xlabel("scan pair index")
    plt.ylabel("ICP final MSE")
    plt.title(f"ICP Convergence Quality over Pairs (dataset {dataset})")
    out = RESULT_DIR / f"scanmatch_mse_dataset{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved mse curve: {out}")

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
