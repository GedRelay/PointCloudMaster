# -*- coding: utf-8 -*-
"""
@Time        :  2024/12/3 19:59
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  carla3
"""
from sceneloader import DatasetLoader_Base
import os
import numpy as np
from utils import Tools

class carla_4d(DatasetLoader_Base):
    def __init__(self, scene_id, json_data):
        super(carla_4d, self).__init__(scene_id, json_data)
        vehicle_state_path = os.path.join(json_data['root_path'], json_data['scenes'][scene_id]['vehicle_state_path'])
        with self.remote.get(vehicle_state_path) as vehicle_state_path:
            self.vehicle_state = self.load_vehicle_state(vehicle_state_path)
        pose_path = os.path.join(json_data['root_path'], json_data['scenes'][scene_id]['pose_path'])
        with self.remote.get(pose_path) as pose_path:
            self.Rs, self.Ts = self.load_poses(pose_path)


    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return: pcd_xyz, 当前帧的点云数据
        :return: other_data, 当前帧的其他数据
        '''
        pcd_path = os.path.join(self.pcd_data_path, self.filenames[frame_id])
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('cos_angle', np.float32),
            ('v', np.float32),
            ('vx', np.float32),
            ('vy', np.float32),
            ('vz', np.float32),
            ('v_cps', np.float32),
            ('intensity', np.float32),
            ('id', np.uint32),
            ('label', np.uint32)
        ])
        with self.remote.get(pcd_path) as pcd_path:
            data = np.fromfile(pcd_path, dtype=dtype)
        pcd_xyz = np.stack([data['x'], data['y'], data['z']], axis=1)
        other_data = {}
        other_data['pointinfo-cos_angle'] = data['cos_angle']
        other_data['pointinfo-rv'] = data['v']
        other_data['pointinfo-real_v'] = np.stack([data['vx'], data['vy'], data['vz']], axis=1)
        other_data['pointinfo-vcps'] = data['v_cps']
        other_data['pointinfo-intensity'] = data['intensity']
        other_data['pointinfo-id'] = data['id']
        other_data['pointinfo-label'] = data['label']
        other_data['vehicle-real_v'] = self.vehicle_state['vehicle-real_v'][frame_id]
        other_data['vehicle-acc'] = self.vehicle_state['vehicle-acc'][frame_id]
        other_data['pose-R'] = self.Rs[frame_id]
        other_data['pose-T'] = self.Ts[frame_id]

        return pcd_xyz, other_data

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
            R = Tools.euler2mat(roll, pitch, yaw)
            T = pose[2:5].astype(np.float32)
            Rs.append(R)
            Ts.append(T)

        return Rs, Ts

    def load_vehicle_state(self, vehicle_state_path):
        '''
        获取所有帧的车辆状态
        :param vehicle_state_path: 车辆状态文件路径
        :return: vehicle_state, 字典，包含车辆状态信息
        '''
        import csv

        vehicle_state = {}
        data = []

        with open(vehicle_state_path, 'r') as f:
            reader = csv.reader(f)
            _ = next(reader)
            for row in reader:
                if row:
                    # frame, timestamp, x, y, z, roll, pitch, yaw, speed, vx, vy, vz, ax, ay, az, throttle, brake, steer, reverse, gear
                    data.append(row)

        data = np.array(data)
        real_v = data[:, 9:12].astype(np.float32)
        acc = data[:, 12:15].astype(np.float32)

        vehicle_state['vehicle-real_v'] = real_v
        vehicle_state['vehicle-acc'] = acc

        return vehicle_state