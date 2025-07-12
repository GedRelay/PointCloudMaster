# -*- coding: utf-8 -*-
from core import DatasetBase
import os
import numpy as np

class aeva(DatasetBase):
    def __init__(self, scene_id, dataset_config):
        super().__init__(scene_id, dataset_config)
        self.pcd_filenames = self.remote.listdir(self.pcd_path)
        self.pcd_filenames.sort(key=lambda x: int(x.split('.')[0]))
        with self.remote.get(self.pose_path) as pose_path:
            self.Rs, self.Ts = self.load_poses(pose_path)

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return: frame_data: 当前帧的数据
        '''
        with self.remote.get(os.path.join(self.pcd_path, self.pcd_filenames[frame_id])) as pcd_path:
            # x, y, z, rv, time
            data = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)
        self.frame_data.pcd.points = data[:, :3]
        self.frame_data.pcd.v_r = data[:, 3]
        self.frame_data.pcd.time = data[:, 4]
        self.frame_data.pose.R = self.Rs[frame_id]
        self.frame_data.pose.T = self.Ts[frame_id]
        return self.frame_data

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