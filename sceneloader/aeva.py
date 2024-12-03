# -*- coding: utf-8 -*-
"""
@Time        :  2024/12/3 17:33
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  datasetloader_aeva
"""
from sceneloader import DatasetLoader_Base
import os
import numpy as np

class aeva(DatasetLoader_Base):
    def __init__(self, scene_id, json_data):
        super(aeva, self).__init__(scene_id, json_data)

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return:
        '''
        # x, y, z, rv, time
        data = np.fromfile(os.path.join(self.pcd_data_path, self.filenames[frame_id]), dtype=np.float32).reshape(-1, 5)
        pcd_xyz = data[:, :3]
        other_data = {}
        other_data['pointinfo-rv'] = data[:, 3]
        other_data['pointinfo-time'] = data[:, 4]

        return pcd_xyz, other_data

    def load_poses(self, scene_id):
        '''
        获取所有帧的位姿
        :param scene_id: 场景id
        :return: Rs, 旋转矩阵列表 [N, 3, 3]
        :return: Ts, 平移向量列表 [N, 3]
        '''

        poses = np.loadtxt(self.pose_path).reshape(-1, 3, 4)

        Rs = poses[:, :3, :3]
        Ts = poses[:, :3, 3]

        return Rs, Ts