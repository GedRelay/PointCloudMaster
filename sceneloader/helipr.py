# -*- coding: utf-8 -*-
"""
@Time        :  2024/12/3 17:34
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  datasetloader_helipr
"""
from sceneloader import DatasetLoader_Base
import os
import numpy as np

class helipr(DatasetLoader_Base):
    def __init__(self, scene_id, json_data):
        super(helipr, self).__init__(scene_id, json_data)

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return:
        '''
        # x, y, z, reflectivity, vcps, time, line_index, intensity
        dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('reflectivity', np.float32),
                          ('velocity', np.float32), ('time_offset_ns', np.int32), ('line_index', np.uint8),
                          ('intensity', np.float32)])
        with self.remote.get(os.path.join(self.pcd_data_path, self.filenames[frame_id])) as f:
            data = np.array(np.fromfile(f, dtype=dtype).tolist())
        pcd_xyz = data[:, :3]
        other_data = {}
        other_data['pointinfo-vcps'] = data[:, 4]
        other_data['pointinfo-intensity'] = data[:, 7]
        other_data['pointinfo-time'] = data[:, 5]
        other_data['pointinfo-line_index'] = data[:, 6]
        other_data['pointinfo-reflectivity'] = data[:, 3]

        return pcd_xyz, other_data