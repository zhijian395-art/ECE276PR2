import numpy as np

# Lidar pose in robot body frame (meters). Fill from RobotConfiguration.pdf.
# The diagram shows ~298.33 mm forward from the rear axle; assume y=0.
LIDAR_OFFSET_X = 0.29833
LIDAR_OFFSET_Y = 0.0

LIDAR_OFFSET = np.array([LIDAR_OFFSET_X, LIDAR_OFFSET_Y], dtype=float)

# Kinect depth camera intrinsics (from PDF)
KINECT_K = np.array(
    [
        [585.05, 0.0, 242.94],
        [0.0, 585.05, 315.84],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

# Kinect depth camera extrinsics in robot body frame (meters, radians)
KINECT_T = np.array([0.18, 0.005, 0.36], dtype=float)
KINECT_RPY = np.array([0.0, 0.36, 0.021], dtype=float)
