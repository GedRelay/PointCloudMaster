# -*- coding: utf-8 -*-
"""
@Time        :  2024/12/3 17:32
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  datasetloader_carla2
"""
from sceneloader import DatasetLoader_Base
import os
import numpy as np
from utils import Tools

class carla2(DatasetLoader_Base):
    def __init__(self, scene_id, json_data):
        super(carla2, self).__init__(scene_id, json_data)


    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return: pcd_xyz, 当前帧的点云数据
        :return: other_data, 当前帧的其他数据
        '''
        # x, y, z, cosangle, rv, vx, vy, vz, vcps, id, label, intensity
        data = np.load(os.path.join(self.pcd_data_path, self.filenames[frame_id]))
        pcd_xyz = data[:, :3]
        other_data = {}
        other_data['pointinfo-rv'] = data[:, 4]
        other_data['pointinfo-vcps'] = data[:, 8]
        other_data['pointinfo-id'] = data[:, 9]
        other_data['pointinfo-label'] = data[:, 10]
        other_data['pointinfo-intensity'] = data[:, 11]
        other_data['pointinfo-cosangle'] = data[:, 3]

        return pcd_xyz, other_data

    def load_poses(self, scene_id):
        '''
        获取所有帧的位姿
        :param scene_id: 场景id
        :return: Rs, 旋转矩阵列表 [N, 3, 3]
        :return: Ts, 平移向量列表 [N, 3]
        '''
        import csv

        poses = []
        with open(self.pose_path, 'r') as f:
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
            R = Tools.euler2mat(roll, pitch, yaw)
            T = pose[2:5].astype(np.float32)
            Rs.append(R)
            Ts.append(T)

        return Rs, Ts