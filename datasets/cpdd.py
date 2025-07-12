# -*- coding: utf-8 -*-
from core import DatasetBase, Tools
import os
import numpy as np
import pickle
import io

from PIL import Image
import json
import math


class cpdd(DatasetBase):
    def __init__(self, scene_id, dataset_config):
        super(cpdd, self).__init__(scene_id, dataset_config)
        self.pcd_filenames = self.remote.listdir(self.pcd_path)
        self.image_filenames = self.remote.listdir(self.image_path)
        self.labels_filenames = self.remote.listdir(self.label_path)
        self.pose_filenames = self.remote.listdir(self.pose_path)  


    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return:
        '''
        with self.remote.get(os.path.join(self.pcd_path, self.pcd_filenames[frame_id])) as pcd_path:
            # x, y, z, reflectivity, velocity, time-offset, line-index, intensity
            data = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 8)

        self.frame_data.frame_id = frame_id
        self.frame_data.pcd.points = data[:, :3]
        self.frame_data.pcd.reflectivity = data[:, 3]
        self.frame_data.pcd.velocity = data[:, 4]
        self.frame_data.pcd.time_offset = data[:, 5]
        self.frame_data.pcd.line_index = data[:, 6]
        self.frame_data.pcd.intensity = data[:, 7]

        # with self.remote.get(self.calib_path) as calib_path:
        #     self.frame_data.calib = self.load_calib(calib_path)

        with self.remote.get(os.path.join(self.image_path, self.image_filenames[frame_id])) as img_path:
            self.frame_data.image = load_image_to_memory(img_path)

        with self.remote.get(os.path.join(self.label_path, self.labels_filenames[frame_id])) as label_path:
            self.frame_data.bbox_3d_corners = self.load_bbox(label_path)

        with self.remote.get(os.path.join(self.pose_path, self.pose_filenames[frame_id])) as pose_path:
            self.frame_data.pose.R, self.frame_data.pose.T = self.load_pose(pose_path)

        return self.frame_data
    
    def parse_inference_bbox(self, inference_box):
        # inference_box: (n, 7) -> [x, y, z, l, w, h, yaw]
        bboxes_corners = []
        for box in inference_box:
            x, y, z, l, w, h, yaw = box
            position = np.array([x, y, z])
            scale = np.array([l, w, h])
            rotation = np.array([0, 0, yaw])
            bbox_corners = get_bbox_corners_by_psr(position, rotation, scale)
            bboxes_corners.append(bbox_corners)
        return bboxes_corners

    def load_bbox(self, label_path):
        with open(label_path, 'r') as file:
            label_data = json.load(file)
        # 计算出包围盒的8个角点，返回列表
        bboxes_corners = []
        for obj in label_data:
            # obj_id = int(obj['obj_id'])
            position = np.array([obj['psr']['position']['x'], obj['psr']['position']['y'], obj['psr']['position']['z']])
            rotation = np.array([obj['psr']['rotation']['x'], obj['psr']['rotation']['y'], obj['psr']['rotation']['z']])
            scale = np.array([obj['psr']['scale']['x'], obj['psr']['scale']['y'], obj['psr']['scale']['z']])
            bbox_corners = get_bbox_corners_by_psr(position, rotation, scale)
            bboxes_corners.append(bbox_corners)
        return bboxes_corners

    def load_pose(self, pose_path):
        with open(pose_path, 'r') as file:
            pose_data = json.load(file)
        roll, pitch, yaw = pose_data["roll"], pose_data["pitch"], pose_data["azimuth"]
        x, y, z = pose_data["x"], pose_data["y"], pose_data["z"]
        R = Tools.Math.euler2mat(float(roll), float(pitch), float(yaw))
        T = np.array([float(x), float(y), float(z)])
        return R, T


def get_bbox_corners_by_psr(position, rotation, scale):
    # 计算包围盒的8个角点
    x, y, z = position
    l, w, h = scale
    _, _, yaw = rotation
    # 旋转矩阵
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    # 8个角点
    x_corners = [l / 2, l / 2, l / 2, l / 2, -l / 2, -l / 2, -l / 2, -l / 2]
    y_corners = [-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2]
    z_corners = [-h / 2, h / 2, h / 2, -h / 2, -h / 2, h / 2, h / 2, -h / 2]
    # 旋转和平移
    corners = np.array([x_corners, y_corners, z_corners])
    corners = np.dot(R, corners)
    corners += np.vstack([x, y, z])
    # 转换为列表
    return corners.T.tolist()


def load_image_to_memory(img_path):
    with open(img_path, 'rb') as file:
        img_data = file.read()
    image = Image.open(io.BytesIO(img_data))
    return image
