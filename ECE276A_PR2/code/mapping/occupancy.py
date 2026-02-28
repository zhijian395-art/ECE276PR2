import numpy as np


def init_map(res=0.05, xy_min=(-20.0, -20.0), xy_max=(20.0, 20.0)):
    res = np.array([res, res], dtype=float)
    xy_min = np.array(xy_min, dtype=float)
    xy_max = np.array(xy_max, dtype=float)
    size = np.ceil((xy_max - xy_min) / res).astype(int)
    is_even = size % 2 == 0
    size[is_even] += 1
    grid = np.zeros(size, dtype=float)
    return {"res": res, "min": xy_min, "max": xy_max, "size": size, "logodds": grid}


def world_to_map(MAP, xy):
    return np.floor((xy - MAP["min"]) / MAP["res"]).astype(int)


def in_bounds(MAP, cells):
    return np.all((cells >= 0) & (cells < MAP["size"]), axis=1)


def bresenham2d(sx, sy, ex, ey):
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex - sx)
    dy = abs(ey - sy)
    steep = abs(dy) > abs(dx)
    if steep:
        dx, dy = dy, dx
    if dy == 0:
        q = np.zeros((dx + 1, 1))
    else:
        q = np.append(
            0,
            np.greater_equal(
                np.diff(np.mod(np.arange(np.floor(dx / 2), -dy * dx + np.floor(dx / 2) - 1, -dy), dx)),
                0,
            ),
        )
    if steep:
        if sy <= ey:
            y = np.arange(sy, ey + 1)
        else:
            y = np.arange(sy, ey - 1, -1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx, ex + 1)
        else:
            x = np.arange(sx, ex - 1, -1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x, y)).T.astype(int)


def update_occupancy(
    MAP,
    origin_xy,
    points_xy,
    lo_occ=0.85,
    lo_free=0.4,
    lo_max=5.0,
    lo_min=-5.0,
):
    origin_cell = world_to_map(MAP, np.asarray(origin_xy))
    pts_cell = world_to_map(MAP, points_xy)
    valid = in_bounds(MAP, pts_cell)
    pts_cell = pts_cell[valid]

    for cell in pts_cell:
        ray = bresenham2d(origin_cell[0], origin_cell[1], cell[0], cell[1])
        # filter to in-bounds cells
        valid_ray = (ray[:, 0] >= 0) & (ray[:, 0] < MAP["size"][0]) & (ray[:, 1] >= 0) & (ray[:, 1] < MAP["size"][1])
        ray = ray[valid_ray]
        if ray.shape[0] == 0:
            continue
        free_cells = ray[:-1]
        occ_cell = ray[-1]
        MAP["logodds"][free_cells[:, 0], free_cells[:, 1]] -= lo_free
        MAP["logodds"][occ_cell[0], occ_cell[1]] += lo_occ

    MAP["logodds"] = np.clip(MAP["logodds"], lo_min, lo_max)


def logodds_to_prob(logodds):
    return 1.0 / (1.0 + np.exp(-logodds))


def threshold_map(prob, occ_thresh=0.65, free_thresh=0.35):
    """
    Returns int8 map: 1 occupied, 0 unknown, -1 free.
    """
    out = np.zeros_like(prob, dtype=np.int8)
    out[prob >= occ_thresh] = 1
    out[prob <= free_thresh] = -1
    return out
