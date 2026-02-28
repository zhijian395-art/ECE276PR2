import numpy as np


def _bruteforce_nn(src, dst, chunk_size=256):
    """
    Brute-force nearest neighbors with chunking to limit memory.
    src: (N, D), dst: (M, D)
    Returns indices (N,) and squared distances (N,).
    """
    n = src.shape[0]
    idx = np.zeros(n, dtype=np.int64)
    d2 = np.zeros(n, dtype=np.float64)
    for i in range(0, n, chunk_size):
        s = src[i : i + chunk_size]
        # (B, 1, D) - (1, M, D) -> (B, M, D)
        diff = s[:, None, :] - dst[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        nn = np.argmin(dist2, axis=1)
        idx[i : i + s.shape[0]] = nn
        d2[i : i + s.shape[0]] = dist2[np.arange(s.shape[0]), nn]
    return idx, d2


def nearest_neighbors(src, dst, max_dist=None, chunk_size=256):
    """
    Compute nearest neighbors in dst for each point in src.
    If SciPy is available, use cKDTree; otherwise use chunked brute-force.
    Returns (src_idx, dst_idx, dist).
    """
    if src.size == 0 or dst.size == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )

    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(dst)
        dist, idx = tree.query(src, k=1)
        if max_dist is not None:
            mask = dist <= max_dist
            src_idx = np.nonzero(mask)[0]
            return src_idx, idx[mask], dist[mask]
        src_idx = np.arange(src.shape[0], dtype=np.int64)
        return src_idx, idx, dist
    except Exception:
        idx, dist2 = _bruteforce_nn(src, dst, chunk_size=chunk_size)
        dist = np.sqrt(dist2)
        if max_dist is not None:
            mask = dist <= max_dist
            src_idx = np.nonzero(mask)[0]
            return src_idx, idx[mask], dist[mask]
        src_idx = np.arange(src.shape[0], dtype=np.int64)
        return src_idx, idx, dist
