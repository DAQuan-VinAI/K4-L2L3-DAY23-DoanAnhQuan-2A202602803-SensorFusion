"""Numerical settings for the teaching EKF in SI units and image pixels.

The state has three position and three velocity coordinates. An observation
history of six samples represents 0.6 s at 10 Hz. Existing lab thresholds and
noise magnitudes are retained to keep student experiments comparable; these are
teaching settings, not fitted sensor uncertainty estimates.
"""

# Kinematics: position followed by velocity; simplified diagonal process noise.
dim_state = 2 * 3
dt = 1.0 / 10.0
q = 3.0

# Independent measurement errors in metres and pixels, respectively.
sigma_lidar_x, sigma_lidar_y, sigma_lidar_z = (0.1,) * 3
sigma_cam_i, sigma_cam_j = (5.0,) * 2

# Broad initial velocity uncertainty: horizontal axes, then vertical axis.
sigma_p44, sigma_p55, sigma_p66 = (50.0, 50.0, 5.0)

# Existence policy applied only by the lidar pass.
window = 6
confirmed_threshold, delete_threshold = (0.8, 0.6)
max_P = 3.0**2

# Shape exponential averaging and probability mass retained by the chi-square gate.
weight_dim = 0.1
gating_threshold = 0.995
