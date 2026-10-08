"""Shared constants for the lab's six-state vehicle tracker."""

# Motion model and lifecycle settings.
dim_state = 6
dt = 0.1
q = 3
confirmed_threshold = 0.8
delete_threshold = 0.6
window = 6
max_P = 9
sigma_p44 = 50
sigma_p55 = 50
sigma_p66 = 5
weight_dim = 0.1
gating_threshold = 0.995

# Measurement standard deviations (metres for lidar, pixels for camera).
sigma_lidar_x = 0.1
sigma_lidar_y = 0.1
sigma_lidar_z = 0.1
sigma_cam_i = 5
sigma_cam_j = 5
