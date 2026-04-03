# -*- coding: utf-8 -*-
from core import DatasetBase
import os
import numpy as np
import json
import cv2

class aevascenes(DatasetBase):
    def __init__(self, scene_id, dataset_config):
        super().__init__(scene_id, dataset_config)

        all_filenames = self.remote.listdir(self.pcd_path)
        self.timestamps = sorted(set(int(f.split('_')[-1].split('.')[0]) for f in all_filenames))  # 时间戳列表提取后去重排序
        self.frame_num = len(self.timestamps)

        with self.remote.get(self.sequence_path) as sequence_path:
            with open(sequence_path, 'r') as f:
                jsondata = json.load(f)
                self.sequence_framedata = jsondata['frames']
                self.sequence_metadata = jsondata['metadata']
        
        self.calib = self.load_calibration()


    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return: frame_data: 当前帧的数据
        '''
        self.frame_data.sequence_uuid = self.sequence_metadata['sequence_uuid']  # 序列uuid
        self.frame_data.frame_uuid = self.sequence_framedata[frame_id]['frame_uuid']  # 帧uuid
        self.frame_data.timestamp = self.timestamps[frame_id]  # 时间戳
        self.frame_data.calib = self.calib  # 校准参数

        points = []
        intensity = []
        velocity = []
        line_index = []
        time_offset_ns = []
        semantic_labels = []
        semantic_labels_idx = []
        for lidar in ['front_narrow_lidar', 'front_wide_lidar', 'left_lidar','rear_narrow_lidar', 'rear_wide_lidar', 'right_lidar']:
            filename = f"{lidar}_{self.timestamps[frame_id]}.npz"
            with self.remote.get(os.path.join(self.pcd_path, filename)) as pcd_path:
                with np.load(pcd_path, allow_pickle=True) as data:
                    points.append(self.pcd_lidar_to_vehicle(data['xyz'], self.sequence_metadata['vehicle_to_lidar_extrinsics'][lidar]))
                    intensity.append(data['reflectivity'])
                    velocity.append(data['velocity'])
                    line_index.append(data['line_index'])
                    time_offset_ns.append(data['time_offset_ns'])
                    semantic_labels.append(data['semantic_labels'])
                    semantic_labels_idx.append(data['semantic_labels_idx'])

        self.frame_data.pcd.points = np.concatenate(points, axis=0)  # 点云
        self.frame_data.pcd.intensity = np.concatenate(intensity, axis=0)  # 强度
        self.frame_data.pcd.v_r = np.concatenate(velocity, axis=0)  # 径向速度
        self.frame_data.pcd.line_index = np.concatenate(line_index, axis=0)  # 线号
        self.frame_data.pcd.time_offset_ns = np.concatenate(time_offset_ns, axis=0)  # 时间偏移
        self.frame_data.pcd.semantic_labels = np.concatenate(semantic_labels, axis=0)  # 语义标签
        self.frame_data.pcd.semantic_labels_idx = np.concatenate(semantic_labels_idx, axis=0)  # 语义标签索引

        # 加载位姿
        self.frame_data.pose.R, self.frame_data.pose.T = self.load_pose(frame_id)

        # 加载物体信息（物体框，速度箭头，类别名称）
        # 类别：（类别名称为细分类别）
        # Vehicles: car, bus, truck, trailer, vehicle_on_rails, other_vehicle
        # Persons: pedestrian, motorcyclist, bicyclist
        # Objects: bicycle, motorcycle, animal, traffic_item, traffic_sign
        # Structures: pole_trunk, building, other_structure, vegetation
        # Surfaces: road, lane_boundary, road_marking, reflective_markers, sidewalk, other_ground
        self.frame_data.bbox_3d_corners, self.frame_data.velocity_arrows_data, self.frame_data.class_names = self.load_objects(frame_id)

        # 加载图片
        self.frame_data.images = {}
        for camera in ['front_narrow_camera', 'front_wide_camera', 'left_camera','rear_narrow_camera', 'rear_wide_camera', 'right_camera']:
            filename = f"{camera}_{self.timestamps[frame_id]}.jpg"
            with self.remote.get(os.path.join(self.image_path, filename)) as img_path:
                self.frame_data.images[camera] = self.load_image_to_memory(img_path)

        return self.frame_data

    def load_objects(self, frame_id):
        '''
        加载当前帧的物体信息（物体框，速度箭头，类别名称）
        :param frame_id: 帧id
        :return: boxes_corners: 物体框的8个角点列表，每个元素为8x3矩阵
        :return: velocity_arrows_data: 速度箭头数据列表，每个元素为(x,y,z,vx,vy,vz)
        :return: class_names: 物体类别名称列表
        '''
        # 加载boxes并计算8个角点
        boxes_corners = []
        velocity_arrows_data = []  # (x,y,z,vx,vy,vz)的列表
        class_names = []  # 类别名称列表

        for object_info in self.sequence_framedata[frame_id]['boxes']:
            dim = object_info['dimensions']
            translation = object_info['pose']['translation']
            rotation = object_info['pose']['rotation']
            linear_velocity = object_info['linear_velocity']
            cls = object_info['class']
            class_names.append(cls)

            # 1. 计算局部角点
            dx, dy, dz = dim['x'] / 2, dim['y'] / 2, dim['z'] / 2
            local_corners = [
                [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
                [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz]
            ]

            # 2. 应用旋转和平移
            R = self.quaternion_rotation_matrix(rotation['x'], rotation['y'], rotation['z'], rotation['w'])
            tx, ty, tz = translation['x'], translation['y'], translation['z']
            world_corners = []
            for corner in local_corners:
                rotated = np.dot(R, corner)
                world_corners.append([rotated[0] + tx, rotated[1] + ty, rotated[2] + tz])
            # 添加速度箭头数据
            velocity_arrows_data.append([
                translation['x'], translation['y'], translation['z'],
                linear_velocity['x'], linear_velocity['y'], linear_velocity['z']
            ])

            # 3. 转换为8x3矩阵并添加到列表
            matrix_8x3 = np.array(world_corners).reshape(8, 3)
            boxes_corners.append(matrix_8x3)

        return boxes_corners, velocity_arrows_data, class_names

    def pcd_lidar_to_vehicle(self, points, extrinsics):
        '''
        将点云从激光雷达坐标系转换到车辆坐标系
        :param points: 点云数据 [N, 3]
        :param extrinsics: 激光雷达到车辆的外参
        :return: 转换后的点云数据 [N, 3]
        '''
        tx, ty, tz = extrinsics['translation']['x'], extrinsics['translation']['y'], extrinsics['translation']['z']
        qw, qx, qy, qz = extrinsics['rotation']['w'], extrinsics['rotation']['x'], extrinsics['rotation']['y'], extrinsics['rotation']['z']
        R = self.quaternion_rotation_matrix(qx, qy, qz, qw)
        T = np.array([tx, ty, tz])

        # 应用旋转和平移
        points_lidar = points @ R.T + T
        return points_lidar


    def quaternion_rotation_matrix(self, x, y, z, w):
        '''
        通过四元数计算旋转矩阵
        :param x: 四元数x
        :param y: 四元数y
        :param z: 四元数z
        :param w: 四元数w
        :return: R, 旋转矩阵 [3, 3]
        '''
        R = np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
        ])
        return R

    def load_pose(self, frame_id):
        '''
        获取当前帧的位姿
        :return: R, 旋转矩阵 [3, 3]
        :return: T, 平移向量 [3]
        '''
        translation = self.sequence_framedata[frame_id]['ego_pose']['translation']
        rotation = self.sequence_framedata[frame_id]['ego_pose']['rotation']
        tx, ty, tz = translation['x'], translation['y'], translation['z']
        qw, qx, qy, qz = rotation['w'], rotation['x'], rotation['y'], rotation['z']
        T = np.array([tx, ty, tz])
        R = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])
        return R, T

    def load_calibration(self):
        '''
        加载校准参数
        :return: calib, 包含各个传感器的内参和外参
        '''
        vehicle_to_lidar_extrinsics = {}
        for lidar in ['front_narrow_lidar', 'front_wide_lidar', 'left_lidar','rear_narrow_lidar', 'rear_wide_lidar', 'right_lidar']:
            translation = self.sequence_metadata['vehicle_to_lidar_extrinsics'][lidar]['translation']
            rotation = self.sequence_metadata['vehicle_to_lidar_extrinsics'][lidar]['rotation']
            # 转换为4x4矩阵
            tx, ty, tz = translation['x'], translation['y'], translation['z']
            qw, qx, qy, qz = rotation['w'], rotation['x'], rotation['y'], rotation['z']
            R = self.quaternion_rotation_matrix(qx, qy, qz, qw)
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = R
            extrinsic_matrix[:3, 3] = [tx, ty, tz]
            vehicle_to_lidar_extrinsics[lidar] = extrinsic_matrix

        vehicle_to_camera_extrinsics = {}
        for camera in ['front_narrow_camera', 'front_wide_camera', 'left_camera','rear_narrow_camera', 'rear_wide_camera', 'right_camera']:
            translation = self.sequence_metadata['vehicle_to_camera_extrinsics'][camera]['translation']
            rotation = self.sequence_metadata['vehicle_to_camera_extrinsics'][camera]['rotation']
            # 转换为4x4矩阵
            tx, ty, tz = translation['x'], translation['y'], translation['z']
            qw, qx, qy, qz = rotation['w'], rotation['x'], rotation['y'], rotation['z']
            R = self.quaternion_rotation_matrix(qx, qy, qz, qw)
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = R
            extrinsic_matrix[:3, 3] = [tx, ty, tz]
            vehicle_to_camera_extrinsics[camera] = extrinsic_matrix

        camera_intrinsics = {}
        for camera in ['front_narrow_camera', 'front_wide_camera', 'left_camera','rear_narrow_camera', 'rear_wide_camera', 'right_camera']:
            distortion_coefficients = self.sequence_metadata['camera_intrinsics'][camera]['distortion_coefficients']  # len=5
            intrinsic_matrix = self.sequence_metadata['camera_intrinsics'][camera]['matrix']  # len=9
            intrinsic_matrix = np.array(intrinsic_matrix).reshape(3, 3)
            camera_intrinsics[camera] = {
                'intrinsic_matrix': intrinsic_matrix,
                'distortion_coefficients': distortion_coefficients
            }

        calib = {
            'vehicle_to_lidar_extrinsics': vehicle_to_lidar_extrinsics,
            'vehicle_to_camera_extrinsics': vehicle_to_camera_extrinsics,
            'camera_intrinsics': camera_intrinsics
        }

        return calib

    def load_image_to_memory(self, img_path):
        '''
        加载图片到内存
        :param img_path: 图片路径
        :return: image: numpy数组格式的图片数据, shape为(H, W, 3)，格式为BGR
        '''
        image = cv2.imread(img_path)  # 注意OpenCV加载的图像格式为BGR，如果需要RGB格式可以使用cv2.cvtColor进行转换
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image