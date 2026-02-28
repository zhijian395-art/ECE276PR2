import os
import sys
import argparse
from pathlib import Path

import numpy as np

# Avoid matplotlib cache errors in restricted environments.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
from PIL import Image

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
RESULT_DIR = CODE_DIR / "result"
RESULT_DIR.mkdir(exist_ok=True)

from odom.trajectory import integrate_trajectory
from mapping.occupancy import init_map, update_occupancy, logodds_to_prob, threshold_map
from mapping.texture import project_depth_to_points
from mapping.transforms import pose2_to_se3, rpy_to_R, se3_from_r_t
from config import LIDAR_OFFSET, KINECT_K, KINECT_T, KINECT_RPY


def load_mapping_trajectory(dataset, encoder_counts, encoder_stamps, imu_w, imu_stamps):
    traj_file = RESULT_DIR / f"icp_trajectory_dataset{dataset}.npz"
    if traj_file.exists():
        data = np.load(traj_file)
        if "poses_icp" in data and "lidar_stamps" in data:
            poses = data["poses_icp"]
            pose_stamps = data["lidar_stamps"]
            print(f"loaded ICP trajectory: {traj_file}")
            return poses, pose_stamps
    poses = integrate_trajectory(encoder_counts, encoder_stamps, imu_w, imu_stamps)
    print("ICP trajectory not found, fallback to odometry trajectory")
    return poses, encoder_stamps


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
    angle_max = float(lidar["angle_max"])
    angle_inc = float(lidar["angle_increment"])

    return (
        encoder_counts,
        encoder_stamps,
        imu_w,
        imu_stamps,
        lidar_ranges,
        lidar_stamps,
        angle_min,
        angle_max,
        angle_inc,
    )


def load_kinect(dataset):
    base = Path(__file__).resolve().parents[2] / "data"
    kin = np.load(base / f"Kinect{dataset}.npz")
    disp_stamps = kin["disparity_time_stamps"]
    rgb_stamps = kin["rgb_time_stamps"]
    return disp_stamps, rgb_stamps


def _frame_list(dataset, kind):
    base = Path(__file__).resolve().parents[2] / "data" / "dataRGBD" / f"{kind}{dataset}"
    files = list(base.glob(f"{kind.lower()}{dataset}_*.png"))

    def _num(p):
        s = p.stem.split("_")[-1]
        try:
            return int(s)
        except ValueError:
            return 0

    files.sort(key=_num)
    return files


def load_disparity(file_list, idx):
    return np.array(Image.open(file_list[idx]))


def load_rgb(file_list, idx):
    return np.array(Image.open(file_list[idx]))


def ranges_to_points(ranges, angle_min, angle_max, angle_inc):
    angles = np.arange(angle_min, angle_max + 1e-9, angle_inc)
    valid = np.logical_and(ranges > 0.1, ranges < 30.0)
    r = ranges[valid]
    a = angles[valid]
    x = r * np.cos(a)
    y = r * np.sin(a)
    return np.stack([x, y], axis=1)


def nearest_pose_index(stamps, t):
    return int(np.argmin(np.abs(stamps - t)))


def transform_points_2d(points, pose, offset_xy=None):
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    t = np.array([x, y])
    if offset_xy is not None:
        t = t + R @ offset_xy
    return (R @ points.T).T + t


def main():
    parser = argparse.ArgumentParser(description="Part3 mapping")
    parser.add_argument("--dataset", type=int, default=20)
    parser.add_argument("--max_pose_dt", type=float, default=0.05)
    args = parser.parse_args()
    dataset = args.dataset
    (
        encoder_counts,
        encoder_stamps,
        imu_w,
        imu_stamps,
        lidar_ranges,
        lidar_stamps,
        angle_min,
        angle_max,
        angle_inc,
    ) = load_dataset(dataset)

    poses, pose_stamps = load_mapping_trajectory(
        dataset, encoder_counts, encoder_stamps, imu_w, imu_stamps
    )
    MAP = init_map(res=0.05, xy_min=(-20, -20), xy_max=(20, 20))

    # Use a subset of scans for speed
    step = 5
    for i in range(0, lidar_ranges.shape[1], step):
        t = lidar_stamps[i]
        idx = nearest_pose_index(pose_stamps, t)
        pose = poses[idx]

        points = ranges_to_points(lidar_ranges[:, i], angle_min, angle_max, angle_inc)
        points_w = transform_points_2d(points, pose, offset_xy=LIDAR_OFFSET)
        origin_w = transform_points_2d(np.array([[0.0, 0.0]]), pose, offset_xy=LIDAR_OFFSET)[0]

        update_occupancy(MAP, origin_w, points_w)

    # visualize occupancy maps
    logodds = MAP["logodds"]
    prob = logodds_to_prob(logodds)
    thresh = threshold_map(prob)

    # trajectory for overlay
    traj_xy = poses[:, :2]
    traj_cells = (np.floor((traj_xy - MAP["min"]) / MAP["res"])).astype(int)
    inb = (
        (traj_cells[:, 0] >= 0)
        & (traj_cells[:, 0] < MAP["size"][0])
        & (traj_cells[:, 1] >= 0)
        & (traj_cells[:, 1] < MAP["size"][1])
    )
    traj_cells = traj_cells[inb]

    plt.figure(figsize=(6, 6))
    plt.imshow(logodds.T, origin="lower", cmap="gray")
    plt.plot(traj_cells[:, 0], traj_cells[:, 1], "r-", linewidth=0.8)
    plt.title(f"Occupancy Grid (log-odds, dataset {dataset})")
    out = RESULT_DIR / f"occupancy_logodds_dataset{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved occupancy map: {out}")

    plt.figure(figsize=(6, 6))
    plt.imshow(prob.T, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    plt.plot(traj_cells[:, 0], traj_cells[:, 1], "r-", linewidth=0.8)
    plt.title(f"Occupancy Grid (probability, dataset {dataset})")
    out = RESULT_DIR / f"occupancy_prob_dataset{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved probability map: {out}")

    plt.figure(figsize=(6, 6))
    plt.imshow(thresh.T, origin="lower", cmap="gray", vmin=-1, vmax=1)
    plt.plot(traj_cells[:, 0], traj_cells[:, 1], "r-", linewidth=0.8)
    plt.title(f"Occupancy Grid (thresholded, dataset {dataset})")
    out = RESULT_DIR / f"occupancy_threshold_dataset{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved threshold map: {out}")

    # ---- Texture mapping ----
    disp_stamps, rgb_stamps = load_kinect(dataset)
    disp_files = _frame_list(dataset, "Disparity")
    rgb_files = _frame_list(dataset, "RGB")
    tex = np.zeros((MAP["size"][0], MAP["size"][1], 3), dtype=float)
    cnt = np.zeros((MAP["size"][0], MAP["size"][1]), dtype=float)

    # camera extrinsics
    R_cb = rpy_to_R(KINECT_RPY[0], KINECT_RPY[1], KINECT_RPY[2])
    T_cb = se3_from_r_t(R_cb, KINECT_T)
    # Depth points are in optical frame (x right, y down, z forward).
    # Convert to body-like camera frame (x forward, y left, z up) before camera extrinsics.
    R_opt_to_cam = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )

    # sample frames for speed
    frame_step = 10
    for di in range(0, disp_stamps.shape[0], frame_step):
        t = disp_stamps[di]
        pose_idx = nearest_pose_index(pose_stamps, t)
        if abs(float(pose_stamps[pose_idx]) - float(t)) > args.max_pose_dt:
            continue
        pose = poses[pose_idx]

        # find nearest RGB frame
        rgb_idx = int(np.argmin(np.abs(rgb_stamps - t)))

        if di >= len(disp_files) or rgb_idx >= len(rgb_files):
            continue
        disp = load_disparity(disp_files, di)
        rgb = load_rgb(rgb_files, rgb_idx)

        pts_c, cols = project_depth_to_points(disp, rgb, KINECT_K, stride=4)
        pts_c = (R_opt_to_cam @ pts_c.T).T

        # transform to world
        T_wb = pose2_to_se3(pose)
        T_wc = T_wb @ T_cb
        ones = np.ones((pts_c.shape[0], 1))
        pts_h = np.hstack([pts_c, ones])
        pts_w = (T_wc @ pts_h.T).T[:, :3]

        # keep floor points only
        z = pts_w[:, 2]
        mask = (z > -0.1) & (z < 0.1)
        pts_w = pts_w[mask]
        cols = cols[mask]

        cells = (np.floor((pts_w[:, :2] - MAP["min"]) / MAP["res"])).astype(int)
        inb = (
            (cells[:, 0] >= 0)
            & (cells[:, 0] < MAP["size"][0])
            & (cells[:, 1] >= 0)
            & (cells[:, 1] < MAP["size"][1])
        )
        cells = cells[inb]
        cols = cols[inb]

        for (cx, cy), col in zip(cells, cols):
            tex[cx, cy] = (tex[cx, cy] * cnt[cx, cy] + col) / (cnt[cx, cy] + 1.0)
            cnt[cx, cy] += 1.0

    # save texture map
    tex_img = tex.astype(np.uint8)
    plt.figure(figsize=(6, 6))
    plt.imshow(tex_img.transpose(1, 0, 2), origin="lower")
    plt.title(f"Texture Map (floor, dataset {dataset})")
    out = RESULT_DIR / f"texture_map_dataset{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved texture map: {out}")

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
