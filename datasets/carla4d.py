# -*- coding: utf-8 -*-
import os
import numpy as np
from core import DatasetBase, Tools

class carla4d(DatasetBase):
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
        filename = self.pcd_filenames[frame_id]

        with self.remote.get(os.path.join(self.pcd_path, filename)) as pcd_path:
            # x, y, z, v_r, intensity
            data = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)

        self.frame_data.pcd.points = data[:, :3]
        self.frame_data.pcd.v_r = data[:, 3]
        self.frame_data.pcd.intensity = data[:, 4]


        with self.remote.get(os.path.join(self.velocity_path, filename)) as f:
            # v_x, v_y, v_z
            velocity_data = np.fromfile(f, dtype=np.float32).reshape(-1, 3)
        self.frame_data.pcd.velocity = velocity_data

        with self.remote.get(os.path.join(self.segmentation_path, filename)) as f:
            # id, label
            data = np.fromfile(f, dtype=np.float32).reshape(-1, 2)
        
        self.frame_data.pcd.id = data[:, 0].astype(np.int32)
        self.frame_data.pcd.label = data[:, 1].astype(np.int32)

        self.frame_data.pose.R = self.Rs[frame_id]
        self.frame_data.pose.T = self.Ts[frame_id]

        return self.frame_data

    def load_poses(self, pose_path):
        '''
        获取所有帧的位姿
        :param pose_path: 位姿文件路径
        :return: Rs, 旋转矩阵列表 [N, 3, 3]
        :return: Ts, 平移向量列表 [N, 3]
        '''
        import csv

        poses = []
        with open(pose_path, 'r') as f:
            reader = csv.reader(f)
            _ = next(reader)
            for row in reader:
                if row:
                    # frame, timestamp, x, y, z, roll, pitch, yaw
                    poses.append(row)
        poses = np.array(poses)
        Rs = []
        Ts = []

        for pose in poses:
            roll, pitch, yaw = pose[5:8].astype(np.float32)
            R = Tools.Math.euler2mat(roll, pitch, yaw)
            T = pose[2:5].astype(np.float32)
            Rs.append(R)
            Ts.append(T)

        return Rs, Ts
