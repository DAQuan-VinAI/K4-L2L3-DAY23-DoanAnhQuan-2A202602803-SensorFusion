import numpy as np
from easydict import EasyDict as edict


def test_bev_maps_shape(workspace_modules):
    bev = workspace_modules["bev_mapping"]
    configs = edict(
        lim_x=[0, 50],
        lim_y=[-25, 25],
        lim_z=[-1, 3],
        bev_width=608,
        bev_height=608,
    )
    pcl = np.random.rand(100, 4).astype(np.float32)
    pcl[:, 0] *= 40
    pcl[:, 1] = pcl[:, 1] * 50 - 25
    pcl[:, 2] = pcl[:, 2] * 2
    maps = bev.bev_maps_from_pcl(pcl, configs)
    assert maps.shape == (3, 608, 608)
