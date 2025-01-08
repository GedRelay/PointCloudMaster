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
        # 将self.filenames中的文件名按照时间戳排序
        self.filenames = sorted(self.filenames, key=lambda x: int(x.split('__')[-1].split('.')[0]))

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return:
        '''

        # x, y, z, intensity, timestamp
        with self.remote.get(os.path.join(self.pcd_data_path, self.filenames[frame_id])) as pcd_path:
            data = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)
        pcd_xyz = data[:, :3]
        other_data = {}
        other_data['pointinfo-intensity'] = data[:, 3]
        other_data['pointinfo-time'] = data[:, 4]

        return pcd_xyz, other_data