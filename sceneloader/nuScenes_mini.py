# -*- coding: utf-8 -*-
"""
@Time        :  2024/12/10 21:25
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  nuScenes_mini
"""
from sceneloader import DatasetLoader_Base
import os
import numpy as np

class nuScenes_mini(DatasetLoader_Base):
    def __init__(self, scene_id, json_data):
        super(nuScenes_mini, self).__init__(scene_id, json_data)

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return:
        '''

        # x, y, z, intensity, timestamp
        data = np.fromfile(os.path.join(self.pcd_data_path, self.filenames[frame_id]), dtype=np.float32).reshape(-1, 5)
        pcd_xyz = data[:, :3]
        other_data = {}
        other_data['pointinfo-intensity'] = data[:, 3]
        other_data['pointinfo-time'] = data[:, 4]

        return pcd_xyz, other_data