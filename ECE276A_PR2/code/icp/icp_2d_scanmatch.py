import numpy as np

from .correspondences import nearest_neighbors
from .se2_fit import fit_se2, se2_to_mat


def transform_points_2d(pts, T):
    ones = np.ones((pts.shape[0], 1))
    hom = np.hstack([pts, ones])
    out = (T @ hom.T).T
    return out[:, :2]


def icp_2d(
    src,
    dst,
    init_T=None,
    max_iters=30,
    tol=1e-4,
    max_dist=0.5,
    min_inliers=20,
):
    """
    Simple point-to-point ICP in 2D.
    src, dst: (N,2), (M,2)
    init_T: 3x3 initial transform (src -> dst)
    Returns T, history (mse list).
    """
    if init_T is None:
        T = np.eye(3)
    else:
        T = init_T.copy()

    prev_mse = None
    history = []

    for _ in range(max_iters):
        src_t = transform_points_2d(src, T)
        src_idx, dst_idx, dist = nearest_neighbors(src_t, dst, max_dist=max_dist)
        if src_idx.size < min_inliers:
            break

        dst_match = dst[dst_idx]
        src_match = src_t[src_idx]

        R, t = fit_se2(src_match, dst_match)
        dT = se2_to_mat(R, t)
        T = dT @ T

        mse = float(np.mean(dist ** 2))
        history.append(mse)
        if prev_mse is not None and abs(prev_mse - mse) < tol:
            break
        prev_mse = mse

    return T, history
