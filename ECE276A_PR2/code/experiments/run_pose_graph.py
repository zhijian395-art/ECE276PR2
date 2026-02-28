import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

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

from config import LIDAR_OFFSET, KINECT_K, KINECT_T, KINECT_RPY
from graph.pose_graph import build_pose_graph, optimize_pose_graph, pose_to_T, inv_se2, T_to_pose
from icp.icp_2d_scanmatch import icp_2d
from mapping.occupancy import init_map, update_occupancy, logodds_to_prob, threshold_map
from mapping.texture import project_depth_to_points
from mapping.transforms import pose2_to_se3, rpy_to_R, se3_from_r_t


def load_lidar_dataset(dataset):
    base = Path(__file__).resolve().parents[2] / "data"
    lidar = np.load(base / f"Hokuyo{dataset}.npz")
    return (
        lidar["ranges"],
        lidar["time_stamps"],
        float(lidar["angle_min"]),
        float(lidar["angle_increment"]),
        float(lidar["range_min"]),
        float(lidar["range_max"]),
    )


def load_kinect(dataset=20):
    base = Path(__file__).resolve().parents[2] / "data"
    kin = np.load(base / f"Kinect{dataset}.npz")
    return kin["disparity_time_stamps"], kin["rgb_time_stamps"]


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
    return points_lidar + LIDAR_OFFSET.reshape(1, 2)


def load_icp_trajectory(dataset=20):
    p = RESULT_DIR / f"icp_trajectory_dataset{dataset}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Run run_icp_scanmatch.py first to generate ICP trajectory."
        )
    d = np.load(p)
    poses_init = d["poses_icp"]
    lidar_stamps = d["lidar_stamps"]
    if "rel_icp" in d.files:
        rel_icp = d["rel_icp"]
    else:
        rel = []
        for i in range(poses_init.shape[0] - 1):
            Ti = pose_to_T(poses_init[i])
            Tj = pose_to_T(poses_init[i + 1])
            rel.append(inv_se2(Ti) @ Tj)
        rel_icp = np.array(rel)
    return poses_init, rel_icp, lidar_stamps


def make_loop_candidates(
    poses_init,
    lidar_ranges,
    angle_min,
    angle_inc,
    range_min,
    range_max,
    interval=60,
    mse_thresh=0.06,
    max_loops=120,
    max_loop_dist=4.0,
):
    loop_edges = []
    n = poses_init.shape[0]

    for i in range(0, n - interval, interval):
        j = i + interval
        if np.linalg.norm(poses_init[i, :2] - poses_init[j, :2]) > max_loop_dist:
            continue

        src_l = ranges_to_points_lidar(lidar_ranges[:, i], angle_min, angle_inc, range_min, range_max)
        dst_l = ranges_to_points_lidar(lidar_ranges[:, j], angle_min, angle_inc, range_min, range_max)
        src_b = lidar_to_body(src_l)
        dst_b = lidar_to_body(dst_l)

        # init from current trajectory estimate
        Ti = pose_to_T(poses_init[i])
        Tj = pose_to_T(poses_init[j])
        T_j_i_init = inv_se2(Tj) @ Ti

        T_j_i_icp, hist = icp_2d(src_b, dst_b, init_T=T_j_i_init, max_iters=30, max_dist=0.7)
        if len(hist) == 0:
            continue
        mse = float(hist[-1])
        if mse < mse_thresh:
            # Convert to i->j relative motion for BetweenFactor(Xi, Xj)
            T_i_j = inv_se2(T_j_i_icp)
            loop_edges.append((i, j, T_i_j, mse))
        if len(loop_edges) >= max_loops:
            break

    return loop_edges


def build_occupancy_map(
    poses,
    lidar_ranges,
    angle_min,
    angle_inc,
    range_min,
    range_max,
    res=0.05,
    dataset=20,
):
    return build_occupancy_map_with_snapshots(
        poses,
        lidar_ranges,
        angle_min,
        angle_inc,
        range_min,
        range_max,
        res=res,
        save_dir=None,
        dataset=dataset,
    )


def _save_snapshot_plots(MAP, poses, idx, label, out_dir, dataset=20):
    prob = logodds_to_prob(MAP["logodds"])
    traj_xy = poses[: idx + 1, :2]
    traj_cells = np.floor((traj_xy - MAP["min"]) / MAP["res"]).astype(int)
    inb = (
        (traj_cells[:, 0] >= 0)
        & (traj_cells[:, 0] < MAP["size"][0])
        & (traj_cells[:, 1] >= 0)
        & (traj_cells[:, 1] < MAP["size"][1])
    )
    traj_cells = traj_cells[inb]

    plt.figure(figsize=(6, 6))
    plt.imshow(prob.T, origin="lower", cmap="gray", vmin=0, vmax=1)
    if traj_cells.shape[0] > 0:
        plt.plot(traj_cells[:, 0], traj_cells[:, 1], "r-", linewidth=0.8)
    plt.title(f"Occupancy over time ({label}, dataset {dataset})")
    plt.savefig(out_dir / f"occupancy_time_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved occupancy snapshot at scan {idx}")

    plt.figure(figsize=(6, 6))
    plt.plot(poses[: idx + 1, 0], poses[: idx + 1, 1], "b-", linewidth=1.0)
    plt.axis("equal")
    plt.title(f"Trajectory over time ({label}, dataset {dataset})")
    plt.savefig(out_dir / f"trajectory_time_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()


def build_occupancy_map_with_snapshots(
    poses,
    lidar_ranges,
    angle_min,
    angle_inc,
    range_min,
    range_max,
    res=0.05,
    save_dir=None,
    dataset=20,
):
    MAP = init_map(res=res, xy_min=(-20, -20), xy_max=(20, 20))
    n = min(poses.shape[0], lidar_ranges.shape[1])
    snapshot_labels = ["early", "mid", "final"]
    snapshot_idx = [
        max(0, min(n - 1, int(round(0.2 * (n - 1))))),
        max(0, min(n - 1, int(round(0.6 * (n - 1))))),
        n - 1,
    ]
    snapshot_plan = dict(zip(snapshot_idx, snapshot_labels))

    for i in range(n):
        pose = poses[i]
        pts_l = ranges_to_points_lidar(lidar_ranges[:, i], angle_min, angle_inc, range_min, range_max)
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        t = np.array([x, y]) + R @ LIDAR_OFFSET
        pts_w = (R @ pts_l.T).T + t
        update_occupancy(MAP, t, pts_w)

        if save_dir is not None and i in snapshot_plan:
            _save_snapshot_plots(MAP, poses, i, snapshot_plan[i], save_dir, dataset=dataset)

    return MAP, snapshot_idx


def build_texture_map(
    poses, slam_stamps, dataset=20, map_res=0.05, max_pose_dt=0.05, save_dir=None
):
    MAP = init_map(res=map_res, xy_min=(-20, -20), xy_max=(20, 20))
    tex = np.zeros((MAP["size"][0], MAP["size"][1], 3), dtype=float)
    cnt = np.zeros((MAP["size"][0], MAP["size"][1]), dtype=float)

    disp_stamps, rgb_stamps = load_kinect(dataset)
    disp_files = _frame_list(dataset, "Disparity")
    rgb_files = _frame_list(dataset, "RGB")

    R_cb = rpy_to_R(KINECT_RPY[0], KINECT_RPY[1], KINECT_RPY[2])
    T_cb = se3_from_r_t(R_cb, KINECT_T)
    R_opt_to_cam = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

    frame_step = 10
    n_frames = min(len(disp_stamps), len(disp_files))
    idx_early_tex = int(0.2 * n_frames)
    idx_mid_tex = int(0.6 * n_frames)
    last_loop_di = ((max(1, n_frames) - 1) // frame_step) * frame_step
    saved_early = False
    saved_mid = False
    for di in range(0, n_frames, frame_step):
        t = float(disp_stamps[di])
        pose_idx = nearest_pose_index(slam_stamps, t)
        if abs(float(slam_stamps[pose_idx]) - t) > max_pose_dt:
            continue
        rgb_idx = int(np.argmin(np.abs(rgb_stamps - t)))

        if rgb_idx >= len(rgb_files):
            continue

        disp = load_disparity(disp_files, di)
        rgb = load_rgb(rgb_files, rgb_idx)

        pts_c, cols = project_depth_to_points(disp, rgb, KINECT_K, stride=4)
        if pts_c.shape[0] == 0:
            continue
        pts_c = (R_opt_to_cam @ pts_c.T).T

        T_wb = pose2_to_se3(poses[pose_idx])
        T_wc = T_wb @ T_cb
        ones = np.ones((pts_c.shape[0], 1))
        pts_w = (T_wc @ np.hstack([pts_c, ones]).T).T[:, :3]

        z = pts_w[:, 2]
        mask = (z > -0.1) & (z < 0.1)
        pts_w = pts_w[mask]
        cols = cols[mask]
        if pts_w.shape[0] == 0:
            continue

        cells = np.floor((pts_w[:, :2] - MAP["min"]) / MAP["res"]).astype(int)
        inb = (
            (cells[:, 0] >= 0)
            & (cells[:, 0] < MAP["size"][0])
            & (cells[:, 1] >= 0)
            & (cells[:, 1] < MAP["size"][1])
        )
        cells = cells[inb]
        cols = cols[inb]

        for (cx, cy), c in zip(cells, cols):
            tex[cx, cy] = (tex[cx, cy] * cnt[cx, cy] + c) / (cnt[cx, cy] + 1.0)
            cnt[cx, cy] += 1.0

        if save_dir is not None:
            if di >= idx_early_tex and not saved_early:
                plt.figure(figsize=(6, 6))
                plt.imshow(tex.astype(np.uint8).transpose(1, 0, 2), origin="lower")
                plt.title(f"Texture over time (early, dataset {dataset})")
                plt.savefig(save_dir / "texture_time_early.png", dpi=150, bbox_inches="tight")
                plt.close()
                saved_early = True
                print(f"Saved texture snapshot at frame {di}")

            if di >= idx_mid_tex and not saved_mid:
                plt.figure(figsize=(6, 6))
                plt.imshow(tex.astype(np.uint8).transpose(1, 0, 2), origin="lower")
                plt.title(f"Texture over time (mid, dataset {dataset})")
                plt.savefig(save_dir / "texture_time_mid.png", dpi=150, bbox_inches="tight")
                plt.close()
                saved_mid = True
                print(f"Saved texture snapshot at frame {di}")

            if di == last_loop_di:
                plt.figure(figsize=(6, 6))
                plt.imshow(tex.astype(np.uint8).transpose(1, 0, 2), origin="lower")
                plt.title(f"Texture over time (final, dataset {dataset})")
                plt.savefig(save_dir / "texture_time_final.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Saved texture snapshot at frame {di}")

    return tex.astype(np.uint8)


def save_pose_graph_outputs(poses_icp, poses_opt, loop_edges, MAP_opt, tex_opt, dataset=20):
    out_dir = RESULT_DIR

    np.savez(
        out_dir / f"optimized_trajectory_dataset{dataset}.npz",
        poses_icp=poses_icp,
        poses_opt=poses_opt,
    )

    plt.figure(figsize=(6, 6))
    plt.plot(poses_icp[:, 0], poses_icp[:, 1], "r-", linewidth=1.0, label="ICP")
    plt.plot(poses_opt[:, 0], poses_opt[:, 1], "b-", linewidth=1.0, label="Optimized")
    for i, j, _, _ in loop_edges:
        plt.plot([poses_opt[i, 0], poses_opt[j, 0]], [poses_opt[i, 1], poses_opt[j, 1]], "g-", linewidth=0.3, alpha=0.6)
    plt.axis("equal")
    plt.legend(loc="best")
    plt.title(f"ICP vs Optimized Trajectory (with loop edges, dataset {dataset})")
    plt.savefig(out_dir / "icp_vs_optimized_traj.png", dpi=150, bbox_inches="tight")

    logodds = MAP_opt["logodds"]
    prob = logodds_to_prob(logodds)
    thresh = threshold_map(prob)

    plt.figure(figsize=(6, 6))
    plt.imshow(logodds.T, origin="lower", cmap="gray")
    plt.title(f"Optimized Occupancy (log-odds, dataset {dataset})")
    plt.savefig(out_dir / "occupancy_logodds_opt.png", dpi=150, bbox_inches="tight")

    plt.figure(figsize=(6, 6))
    plt.imshow(prob.T, origin="lower", cmap="gray", vmin=0, vmax=1)
    plt.title(f"Optimized Occupancy (probability, dataset {dataset})")
    plt.savefig(out_dir / "occupancy_prob_opt.png", dpi=150, bbox_inches="tight")

    plt.figure(figsize=(6, 6))
    plt.imshow(thresh.T, origin="lower", cmap="gray", vmin=-1, vmax=1)
    plt.title(f"Optimized Occupancy (threshold, dataset {dataset})")
    plt.savefig(out_dir / "occupancy_threshold_opt.png", dpi=150, bbox_inches="tight")

    plt.figure(figsize=(6, 6))
    plt.imshow(tex_opt.transpose(1, 0, 2), origin="lower")
    plt.title(f"Optimized Texture Map (dataset {dataset})")
    plt.savefig(out_dir / "texture_map_opt.png", dpi=150, bbox_inches="tight")

    try:
        plt.show()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Part4 pose graph optimization with loop closure")
    parser.add_argument("--dataset", type=int, default=20)
    parser.add_argument("--loop_interval", type=int, default=150)
    parser.add_argument("--loop_mse_thresh", type=float, default=0.06)
    parser.add_argument("--max_loops", type=int, default=40)
    parser.add_argument("--loop_max_dist", type=float, default=4.0)
    parser.add_argument("--max_pose_dt", type=float, default=0.05)
    args = parser.parse_args()

    poses_init, rel_icp, lidar_stamps = load_icp_trajectory(args.dataset)
    lidar_ranges, lidar_ts, angle_min, angle_inc, range_min, range_max = load_lidar_dataset(args.dataset)

    n = min(poses_init.shape[0], lidar_ranges.shape[1], lidar_stamps.shape[0])
    poses_init = poses_init[:n]
    rel_icp = rel_icp[: n - 1]
    lidar_stamps = lidar_stamps[:n]
    lidar_ranges = lidar_ranges[:, :n]

    loop_edges = make_loop_candidates(
        poses_init,
        lidar_ranges,
        angle_min,
        angle_inc,
        range_min,
        range_max,
        interval=args.loop_interval,
        mse_thresh=args.loop_mse_thresh,
        max_loops=args.max_loops,
        max_loop_dist=args.loop_max_dist,
    )

    graph, initial = build_pose_graph(
        poses_init,
        rel_icp,
        loop_edges,
        sigma_odom=(0.05, 0.05, 0.02),
        sigma_loop=(0.03, 0.03, 0.01),
        sigma_prior=(1e-4, 1e-4, 1e-4),
    )
    initial_error = float(graph.error(initial))
    poses_opt, result_values = optimize_pose_graph(graph, initial)
    final_error = float(graph.error(result_values))

    MAP_opt, snapshot_idx = build_occupancy_map_with_snapshots(
        poses_opt,
        lidar_ranges,
        angle_min,
        angle_inc,
        range_min,
        range_max,
        save_dir=RESULT_DIR,
        dataset=args.dataset,
    )
    tex_opt = build_texture_map(
        poses_opt,
        lidar_stamps,
        dataset=args.dataset,
        max_pose_dt=args.max_pose_dt,
        save_dir=RESULT_DIR,
    )

    save_pose_graph_outputs(poses_init, poses_opt, loop_edges, MAP_opt, tex_opt, dataset=args.dataset)
    loop_mse = np.array([e[3] for e in loop_edges], dtype=float)
    if loop_mse.size > 0:
        loop_mse_min = float(np.min(loop_mse))
        loop_mse_median = float(np.median(loop_mse))
        loop_mse_max = float(np.max(loop_mse))
    else:
        loop_mse_min = None
        loop_mse_median = None
        loop_mse_max = None

    gap_before = float(np.linalg.norm(poses_init[-1, :2] - poses_init[0, :2]))
    gap_after = float(np.linalg.norm(poses_opt[-1, :2] - poses_opt[0, :2]))

    metrics = {
        "dataset": int(args.dataset),
        "initial_error": initial_error,
        "final_error": final_error,
        "n_loop_edges": int(len(loop_edges)),
        "loop_acceptance_mse_threshold": float(args.loop_mse_thresh),
        "loop_acceptance_max_dist": float(args.loop_max_dist),
        "loop_mse_min": loop_mse_min,
        "loop_mse_median": loop_mse_median,
        "loop_mse_max": loop_mse_max,
        "gap_before": gap_before,
        "gap_after": gap_after,
        "snapshot_indices": [int(x) for x in snapshot_idx],
    }
    metrics_path = RESULT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"initial graph error: {initial_error:.6f}")
    print(f"final graph error:   {final_error:.6f}")
    print(f"loop closures added: {len(loop_edges)}")
    print(
        "loop acceptance: "
        f"mse < {args.loop_mse_thresh}, distance < {args.loop_max_dist} m"
    )
    print(
        "loop mse stats (min/median/max): "
        f"{loop_mse_min} / {loop_mse_median} / {loop_mse_max}"
    )
    print(f"closure gap before/after: {gap_before:.4f} m / {gap_after:.4f} m")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
