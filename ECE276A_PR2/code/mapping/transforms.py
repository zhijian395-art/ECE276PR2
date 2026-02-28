import numpy as np


def rotx(roll):
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roty(pitch):
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_R(roll, pitch, yaw):
    # ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def se3_from_r_t(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pose2_to_se3(pose):
    x, y, theta = pose
    R = rotz(theta)
    t = np.array([x, y, 0.0])
    return se3_from_r_t(R, t)
