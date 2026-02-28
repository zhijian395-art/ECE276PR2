import numpy as np
import gtsam


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
    return np.array([x, y, theta], dtype=float)


def inv_se2(T):
    R = T[:2, :2]
    t = T[:2, 2]
    T_inv = np.eye(3)
    T_inv[:2, :2] = R.T
    T_inv[:2, 2] = -R.T @ t
    return T_inv


def T_to_pose2(T):
    x, y = T[:2, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return gtsam.Pose2(float(x), float(y), float(theta))


def build_pose_graph(
    poses_init,
    rel_icp,
    loop_edges,
    sigma_odom=(0.05, 0.05, 0.02),
    sigma_loop=(0.03, 0.03, 0.01),
    sigma_prior=(1e-4, 1e-4, 1e-4),
):
    """
    Build GTSAM Pose2 graph from ICP trajectory.
    loop_edges: list of (i, j, T_rel_ij, mse)
    """
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    X = lambda k: gtsam.symbol("x", int(k))

    n = poses_init.shape[0]
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(sigma_prior, dtype=float))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(sigma_odom, dtype=float))
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(sigma_loop, dtype=float))

    for i in range(n):
        p = poses_init[i]
        initial.insert(X(i), gtsam.Pose2(float(p[0]), float(p[1]), float(p[2])))

    # Fix origin with prior on first pose.
    p0 = poses_init[0]
    graph.add(gtsam.PriorFactorPose2(X(0), gtsam.Pose2(float(p0[0]), float(p0[1]), float(p0[2])), prior_noise))

    # Consecutive odometry/ICP factors.
    for i in range(n - 1):
        graph.add(gtsam.BetweenFactorPose2(X(i), X(i + 1), T_to_pose2(rel_icp[i]), odom_noise))

    # Loop closure factors.
    for i, j, T_rel_ij, _ in loop_edges:
        graph.add(gtsam.BetweenFactorPose2(X(i), X(j), T_to_pose2(T_rel_ij), loop_noise))

    return graph, initial


def optimize_pose_graph(graph, initial):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    n = initial.size()
    X = lambda k: gtsam.symbol("x", int(k))
    poses_opt = np.zeros((n, 3), dtype=float)
    for i in range(n):
        p = result.atPose2(X(i))
        poses_opt[i] = [p.x(), p.y(), p.theta()]
    return poses_opt, result
