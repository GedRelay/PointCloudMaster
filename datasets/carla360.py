# -*- coding: utf-8 -*-
from core import DatasetBase, Tools
import os
import numpy as np
import pickle
import io
from PIL import Image


class carla360(DatasetBase):
    def __init__(self, scene_id, dataset_config):
        super().__init__(scene_id, dataset_config)
        self.pcd_filenames = self.remote.listdir(self.pcd_path)  # 读取点云文件名
        self.pcd_filenames.sort(key=lambda x: int(x.split('.')[0]))
        self.calib_filenames = self.remote.listdir(self.calib_path)  # 读取标定文件名
        self.calib_filenames.sort(key=lambda x: int(x.split('.')[0]))
        self.image_filenames = self.remote.listdir(self.image_path)  # 读取图片文件名
        self.image_filenames.sort(key=lambda x: int(x.split('.')[0]))
        self.labels_filenames = self.remote.listdir(self.label_path)  # 读取标签文件名
        self.labels_filenames.sort(key=lambda x: int(x.split('.')[0]))
        self.pose_filenames = self.remote.listdir(self.pose_path)  # 读取pose文件名
        self.pose_filenames.sort(key=lambda x: int(x.split('.')[0]))


    def load_frame(self, frame_id):
        with self.remote.get(os.path.join(self.pcd_path, self.pcd_filenames[frame_id])) as pcd_path:
            # x, y, z, cos_angle, rv, vx, vy, vz, vcps, intensity, id, label
            data = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 12)

        self.frame_data.pcd.points = data[:, :3]
        self.frame_data.pcd.cos_angle = data[:, 3]
        self.frame_data.pcd.v_r = data[:, 4]
        self.frame_data.pcd.velocity = data[:, 5:8]
        self.frame_data.pcd.v_cps = data[:, 8]
        self.frame_data.pcd.intensity = data[:, 9]

        with self.remote.get(os.path.join(self.calib_path, self.calib_filenames[frame_id])) as calib_path:
            self.frame_data.calib = self.load_calib(calib_path)
        with self.remote.get(os.path.join(self.image_path, self.image_filenames[frame_id])) as img_path:
            self.frame_data.image = load_image_to_memory(img_path)
        with self.remote.get(os.path.join(self.label_path, self.labels_filenames[frame_id])) as label_path:
            self.frame_data.bbox_2d = self.load_bbox2d(label_path)
            self.frame_data.bbox_3d_corners = self.load_bbox3d(label_path, self.frame_data.calib['R0_inv'],
                                                             self.frame_data.calib['Tr_cam_velo'])
            
        with self.remote.get(os.path.join(self.pose_path, self.pose_filenames[frame_id])) as pose_path:
            self.frame_data.pose.R, self.frame_data.pose.T = self.load_poses(pose_path)

        return self.frame_data

    def load_bbox2d(self, label_path):
        '''
        获取2d bbox
        :param label_path:
        :return:
        '''
        labels = np.loadtxt(label_path, delimiter=' ', dtype=str).reshape(-1, 15)
        bboxes_2d = []
        for label in labels:
            if label[0] == 'DontCare':
                continue
            bbox_2d = label[4:8].astype(np.float32)
            bboxes_2d.append(bbox_2d)
        return bboxes_2d


    def load_bbox3d(self, label_path, R0_inv, Tr_cam_velo):
        '''
        获取3d bbox
        :param label_path:
        :return:
        '''
        labels = np.loadtxt(label_path, delimiter=' ', dtype=str).reshape(-1, 15)
        bboxes_corners = []
        for label in labels:
            if label[0] == 'DontCare':
                continue
            # 获取3d bbox的8个点
            h, w, l, x, y, z, yaw = label[8:15].astype(np.float32)
            corners_3d_cam2 = compute_3d_box_cam2(h, w, l, x, y, z, yaw)  # (3, 8)
            # 将cam坐标系的点转换到velo坐标系
            pts_3d_ref = (R0_inv @ corners_3d_cam2).T
            n = pts_3d_ref.shape[0]
            pts_3d_ref = np.hstack((pts_3d_ref, np.ones((n, 1))))
            corners_3d_velo = pts_3d_ref @ Tr_cam_velo.T  # (8, 3)
            bboxes_corners.append(corners_3d_velo)
        return bboxes_corners


    def load_calib(self, calib_filepath):
        '''
        读取标定文件
        :param calib_filepath:
        :return:
        '''
        calib_ = {}
        calib = {}
        with open(calib_filepath, "r") as f:
            for line in f:
                data = line.strip().split(" ")
                key = data[0]
                value = np.array(data[1:], dtype=np.float32)
                calib_[key] = value
        calib["P0"] = calib_["P0:"].reshape(3, 4)  # 投影矩阵
        calib["P1"] = calib_["P1:"].reshape(3, 4)
        calib["P2"] = calib_["P2:"].reshape(3, 4)
        calib["P2_inv"] = Tools.Math.inverse_rigid_trans(calib["P2"])
        calib["P3"] = calib_["P3:"].reshape(3, 4)
        calib["R0"] = calib_["R0_rect:"].reshape(3, 3)
        calib["R0_inv"] = np.linalg.inv(calib["R0"])
        calib["Tr_velo_cam"] = calib_["Tr_velo_to_cam:"].reshape(3, 4)
        calib["Tr_imu_velo"] = calib_["Tr_imu_to_velo:"].reshape(3, 4)
        calib["Tr_cam_velo"] = Tools.Math.inverse_rigid_trans(calib["Tr_velo_cam"])
        return calib


    def load_poses(self, pose_path):
        '''
        获取lidar位姿
        :param pose_path: 位姿文件路径
        :return: R, 旋转矩阵
        :return: T, 平移向量
        '''
        with open(pose_path, 'r') as f:
            poses = f.readlines()[1]
            # 内容是 "lidar: x y z roll pitch yaw\n"
            # 去除前面的lidar:和后面的\n
            poses = poses.replace('lidar: ', '').replace('\n', '')
            poses = poses.split(" ")
            poses = np.array(poses, dtype=np.float32)

        roll, pitch, yaw = poses[3:].astype(np.float32)
        R = Tools.Math.euler2mat(roll, pitch, yaw)
        T = poses[:3].astype(np.float32)

        return R, T

    def load_pkl(self, pkl_path):
        '''
        读取pkl文件
        :param pkl_path:
        :return:
        '''
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data


def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                  [0, 1, 0],
                  [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2


def load_image_to_memory(img_path):
    with open(img_path, 'rb') as file:
        img_data = file.read()
    image = Image.open(io.BytesIO(img_data))
    return image


