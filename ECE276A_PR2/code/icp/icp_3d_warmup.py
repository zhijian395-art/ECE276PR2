import numpy as np

try:
    from .correspondences import nearest_neighbors
except ImportError:
    # Allow running as a standalone script (not as a package module).
    from correspondences import nearest_neighbors


def fit_se3(src, dst):
    """
    Fit SE(3) transform that maps src -> dst using SVD.
    src, dst: (N, 3)
    Returns R (3x3), t (3,).
    """
    if src.shape[0] == 0:
        return np.eye(3), np.zeros(3)
    mu_s = np.mean(src, axis=0)
    mu_d = np.mean(dst, axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return R, t


def se3_to_mat(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points_3d(pts, T):
    ones = np.ones((pts.shape[0], 1))
    hom = np.hstack([pts, ones])
    out = (T @ hom.T).T
    return out[:, :3]


def _rotz_3d(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def icp_3d(
    src,
    dst,
    init_T=None,
    max_iters=30,
    tol=1e-4,
    max_dist=0.2,
    sample_size=4000,
):
    """
    Simple point-to-point ICP in 3D for warm-up.
    """
    if init_T is None:
        T = np.eye(4)
    else:
        T = init_T.copy()

    # Align rough centroids to help with large translations
    mu_s = np.zeros(3)
    mu_d = np.zeros(3)
    if src.shape[0] > 0 and dst.shape[0] > 0:
        mu_s = np.mean(src, axis=0)
        mu_d = np.mean(dst, axis=0)
        src = src - mu_s
        dst = dst - mu_d

    # Downsample to keep runtime reasonable
    if sample_size is not None:
        if src.shape[0] > sample_size:
            src = src[np.random.choice(src.shape[0], sample_size, replace=False)]
        if dst.shape[0] > sample_size:
            dst = dst[np.random.choice(dst.shape[0], sample_size, replace=False)]

    prev_mse = None
    history = []

    for _ in range(max_iters):
        src_t = transform_points_3d(src, T)
        src_idx, dst_idx, dist = nearest_neighbors(src_t, dst, max_dist=max_dist)
        if src_idx.size == 0:
            break

        dst_match = dst[dst_idx]
        src_match = src_t[src_idx]

        R, t = fit_se3(src_match, dst_match)
        dT = se3_to_mat(R, t)
        T = dT @ T

        mse = float(np.mean(dist ** 2))
        history.append(mse)
        if prev_mse is not None and abs(prev_mse - mse) < tol:
            break
        prev_mse = mse

    # Convert back to original coordinates
    T_full = T.copy()
    T_full[:3, 3] = T[:3, 3] + mu_d - T[:3, :3] @ mu_s
    return T_full, history


def icp_3d_multi_init(src, dst, num_yaw=24, **kwargs):
    """
    Run ICP with yaw-discretized initialization and pick the best result by final MSE.
    """
    best_T = np.eye(4)
    best_hist = []
    best_mse = np.inf

    for yaw in np.linspace(-np.pi, np.pi, num_yaw, endpoint=False):
        T0 = np.eye(4)
        T0[:3, :3] = _rotz_3d(yaw)
        T, hist = icp_3d(src, dst, init_T=T0, **kwargs)
        if len(hist) == 0:
            continue
        mse = hist[-1]
        if mse < best_mse:
            best_mse = mse
            best_T = T
            best_hist = hist

    # Fallback to single-run ICP if all initializations fail.
    if not best_hist:
        best_T, best_hist = icp_3d(src, dst, **kwargs)
    return best_T, best_hist
