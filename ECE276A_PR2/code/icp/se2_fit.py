import numpy as np


def fit_se2(src, dst):
    """
    Fit SE(2) transform that maps src -> dst using SVD.
    src, dst: (N, 2)
    Returns R (2x2), t (2,).
    """
    if src.shape[0] == 0:
        return np.eye(2), np.zeros(2)
    mu_s = np.mean(src, axis=0)
    mu_d = np.mean(dst, axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return R, t


def se2_to_mat(R, t):
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    return T
