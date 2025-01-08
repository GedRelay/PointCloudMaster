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
        self.pose_path = os.path.join(json_data['root_path'], json_data['scenes'][scene_id]['pose_path'])
        with self.remote.get(self.pose_path) as pose_path:
            self.Rs, self.Ts = self.load_poses(pose_path)

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return:
        '''
        # x, y, z, rv, time
        with self.remote.get(os.path.join(self.pcd_data_path, self.filenames[frame_id])) as pcd_file:
            data = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 5)
        pcd_xyz = data[:, :3]
        other_data = {}
        other_data['pointinfo-rv'] = data[:, 3]
        other_data['pointinfo-time'] = data[:, 4]
        other_data['pose-R'] = self.Rs[frame_id]
        other_data['pose-T'] = self.Ts[frame_id]

        return pcd_xyz, other_data

    def load_poses(self, pose_path):
        '''
        获取所有帧的位姿

        :return: Rs, 旋转矩阵列表 [N, 3, 3]
        :return: Ts, 平移向量列表 [N, 3]
        '''
        poses = np.loadtxt(pose_path).reshape(-1, 3, 4)
        Rs = poses[:, :3, :3]
        Ts = poses[:, :3, 3]
        return Rs, Ts