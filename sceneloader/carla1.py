# -*- coding: utf-8 -*-
"""
@Time        :  2024/12/3 17:31
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  datasetloader_carla1
"""
from sceneloader import DatasetLoader_Base
import os
import numpy as np

class carla1(DatasetLoader_Base):
    def __init__(self, scene_id, json_data):
        super(carla1, self).__init__(scene_id, json_data)

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return: pcd_xyz, 当前帧的点云数据
        :return: other_data, 当前帧的其他数据
        '''
        # x, y, z, rv, vx, vy, vz, id, label, intensity
        data = np.load(os.path.join(self.pcd_data_path, self.filenames[frame_id]))
        pcd_xyz = data[:, :3]
        other_data = {}
        other_data['pointinfo-rv'] = data[:, 3]
        other_data['pointinfo-real_v'] = data[:, 4:7]
        other_data['pointinfo-id'] = data[:, 7]
        other_data['pointinfo-intensity'] = data[:, 9]

        return pcd_xyz, other_data