import numpy as np


def disparity_to_depth(d):
    dd = (-0.00304 * d + 3.31)
    return 1.03 / dd


def disparity_to_rgb_coords(i, j, dd):
    rgbi = (526.37 * i + 19276 - 7877.07 * dd) / 585.051
    rgbj = (526.37 * j + 16662) / 585.051
    return rgbi, rgbj


def project_depth_to_points(disparity, rgb, K, stride=4):
    """
    disparity: (H,W) uint16
    rgb: (H,W,3) uint8
    K: 3x3 intrinsics (depth camera)
    Returns points (N,3) and colors (N,3) in camera frame.
    """
    H, W = disparity.shape
    i_idx, j_idx = np.meshgrid(np.arange(W), np.arange(H))

    if stride > 1:
        i_idx = i_idx[::stride, ::stride]
        j_idx = j_idx[::stride, ::stride]
        disparity = disparity[::stride, ::stride]

    d = disparity.astype(np.float64)
    dd = (-0.00304 * d + 3.31)
    depth = 1.03 / dd

    # depth camera coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (i_idx - cx) * depth / fx
    y = (j_idx - cy) * depth / fy
    z = depth

    # RGB pixel mapping
    rgbi, rgbj = disparity_to_rgb_coords(i_idx, j_idx, dd)
    rgbi = np.round(rgbi).astype(int)
    rgbj = np.round(rgbj).astype(int)

    valid = (
        np.isfinite(depth)
        & (dd > 0)
        & (depth > 0)
        & (depth < 20.0)
        & (rgbi >= 0)
        & (rgbi < rgb.shape[1])
        & (rgbj >= 0)
        & (rgbj < rgb.shape[0])
    )

    pts = np.stack([x, y, z], axis=2)[valid]
    cols = rgb[rgbj[valid], rgbi[valid]]
    return pts, cols
