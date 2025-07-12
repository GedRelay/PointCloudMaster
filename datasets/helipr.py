# -*- coding: utf-8 -*-
from core import DatasetBase
import os
import numpy as np

class helipr(DatasetBase):
    def __init__(self, scene_id, dataset_config):
        super().__init__(scene_id, dataset_config)
        self.pcd_filenames = self.remote.listdir(self.pcd_path)
        self.pcd_filenames.sort(key=lambda x: int(x.split('.')[0]))

    def load_frame(self, frame_id):
        # x, y, z, reflectivity, vcps, time, line_index, intensity
        dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('reflectivity', np.float32),
                          ('velocity', np.float32), ('time_offset_ns', np.int32), ('line_index', np.uint8),
                          ('intensity', np.float32)])
        with self.remote.get(os.path.join(self.pcd_path, self.pcd_filenames[frame_id])) as f:
            data = np.array(np.fromfile(f, dtype=dtype).tolist())

        self.frame_data.pcd.points = data[:, :3]
        self.frame_data.pcd.reflectivity = data[:, 3]
        self.frame_data.pcd.v_cps = data[:, 4]
        self.frame_data.pcd.time_offset_ns = data[:, 5]
        self.frame_data.pcd.line_index = data[:, 6]
        self.frame_data.pcd.intensity = data[:, 7]

        return self.frame_data