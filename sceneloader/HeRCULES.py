##########################################
# @Author: rtzhang
# @Date: 2025-03-10 16:57:14
# @LastEditors: rtzhang
# @LastEditTime: 2025-03-10 16:58:53
# @Description: SceneLoader for HeRCULES dataset(unfinished)
##########################################
# -*- coding: utf-8 -*-

from sceneloader import DatasetLoader_Base
import os
import numpy as np
import pickle
import io

from utils import Tools
from PIL import Image

class HeRCULES(DatasetLoader_Base):
    def __init__(self, scene_id, json_data):
        super(HeRCULES, self).__init__(scene_id, json_data)
 

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

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return:
        '''
        pcd_path = os.path.join(self.pcd_data_path, self.filenames[frame_id])

        # uncertain about the data type
        # x, y, z, reflectivity, intensity(float), velocity, line-index(uint8_t), time-offset(int32_t)
        # x, y, z, reflectivity, velocity, time-offset(int32_t), line-index(uint8_t), intensity(float)
        dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),
                            ('reflectivity', np.float32), ('velocity', np.float32),
                            ('time-offset', np.int32), ('line-index', np.uint8), ('intensity', np.float32)])


        with self.remote.get(pcd_path) as pcd_path:
            data = np.fromfile(pcd_path, dtype=dtype)
        pcd_xyz = np.array(data[['x', 'y', 'z']].tolist(), dtype=np.float32)

        other_data = {}
        other_data['pointinfo-velocity'] = np.array(data['velocity'].tolist(), dtype=np.float32)


        return pcd_xyz, other_data

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
        calib["P2_inv"] = Tools.inverse_rigid_trans(calib["P2"])
        calib["P3"] = calib_["P3:"].reshape(3, 4)
        calib["R0"] = calib_["R0_rect:"].reshape(3, 3)
        calib["R0_inv"] = np.linalg.inv(calib["R0"])
        calib["Tr_velo_cam"] = calib_["Tr_velo_to_cam:"].reshape(3, 4)
        calib["Tr_imu_velo"] = calib_["Tr_imu_to_velo:"].reshape(3, 4)
        calib["Tr_cam_velo"] = Tools.inverse_rigid_trans(calib["Tr_velo_cam"])
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
        R = Tools.euler2mat(roll, pitch, yaw)
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


