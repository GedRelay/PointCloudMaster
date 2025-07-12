# -*- coding: utf-8 -*-
from core import DatasetBase, Tools
import os
import numpy as np
from PIL import Image
import io

class kitti_tracking(DatasetBase):
    def __init__(self, scene_id, dataset_config):
        super().__init__(scene_id, dataset_config)
        self.pcd_filenames = self.remote.listdir(self.pcd_path)
        self.pcd_filenames.sort(key=lambda x: int(x.split('.')[0]))
        self.calib = self.load_calib()  # 读取标定文件
        self.image_filenames = self.remote.listdir(self.image_path)  # 读取图片文件名
        self.image_filenames = [f for f in self.image_filenames if f.endswith('.png')]
        self.image_filenames.sort(key=lambda x: int(x.split('.')[0]))
        with self.remote.get(self.label_path) as label_path: # 读取标签文件
            self.labels_data = np.loadtxt(label_path, delimiter=' ', dtype=str)
        self.Rs, self.Ts = self.load_poses(scene_id)  # 读取位姿文件


    def load_frame(self, frame_id):
        with self.remote.get(os.path.join(self.pcd_path, self.pcd_filenames[frame_id])) as pcd_path:
            # x, y, z, intensity, rv, vx, vy, vz
            data = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 8)
        self.frame_data.pcd.points = data[:, :3]
        self.frame_data.pcd.intensity = data[:, 3]
        self.frame_data.pcd.v_r = data[:, 4]
        self.frame_data.pcd.velocity = data[:, 5:8]

        self.frame_data.calib = self.calib

        with self.remote.get(os.path.join(self.image_path, self.image_filenames[frame_id])) as img_path:
            self.frame_data.image = load_image_to_memory(img_path)
        
        self.frame_data.bbox_2d, self.frame_data.bbox_2d_ids = self.load_bbox2d(frame_id)

        self.frame_data.bbox_3d_corners, self.frame_data.bbox_3d_ids = self.load_bbox3d(frame_id)

        self.frame_data.pose.R = self.Rs[frame_id]
        self.frame_data.pose.T = self.Ts[frame_id]

        return self.frame_data

    
    def load_bbox2d(self, frame_id):
        '''
        获取2d bbox
        :param frame_id:
        :return:
        '''
        labels = self.labels_data[(self.labels_data[:, 0] == str(frame_id))]
        bboxes_2d = []
        bboxes_ids = []
        for label in labels:
            if label[2] == 'DontCare':
                continue
            id = label[1]
            bbox_2d = label[6:10].astype(np.float32)
            bboxes_2d.append(bbox_2d)
            bboxes_ids.append(id)
        return bboxes_2d, bboxes_ids


    def load_bbox3d(self, frame_id):
        '''
        获取3d bbox
        :param frame_id:
        :return:
        '''
        labels = self.labels_data[(self.labels_data[:, 0] == str(frame_id))]
        bboxes_corners = []
        bboxes_ids = []
        for label in labels:
            if label[2] == 'DontCare':
                continue
            id = label[1]
            # 获取3d bbox的8个点
            h, w, l, x, y, z, yaw = label[10:17].astype(np.float32)
            corners_3d_cam2 = compute_3d_box_cam2(h, w, l, x, y, z, yaw)  # (3, 8)
            # 将cam坐标系的点转换到velo坐标系
            pts_3d_ref = (self.calib['R0_inv'] @ corners_3d_cam2).T
            n = pts_3d_ref.shape[0]
            pts_3d_ref = np.hstack((pts_3d_ref, np.ones((n, 1))))
            corners_3d_velo = pts_3d_ref @ self.calib['Tr_cam_velo'].T  # (8, 3)
            bboxes_corners.append(corners_3d_velo)
            bboxes_ids.append(id)
        return bboxes_corners, bboxes_ids

    def load_poses(self, scene_id):
        '''
        获取所有帧的位姿
        :param scene_id: 场景id
        :return: Rs, 旋转矩阵列表 [N, 3, 3]
        :return: Ts, 平移向量列表 [N, 3]
        '''

        # lat, lon, alt, roll, pitch, yaw, vn, ve, vf, vl, vu, ax, ay, az, af, al, au, wx, wy, wz, wf, wl, wu, pos_accuracy, vel_accuracy, navstat, posmode, velmode, orimode

        Rs = []
        Ts = []
        scale = None
        origin = None
        with self.remote.get(self.pose_path) as pose_path:
            with open(pose_path, 'r') as f:
                for line in f.readlines():
                    line = line.split()
                    line[:-5] = [float(x) for x in line[:-5]]
                    line[-5:] = [int(float(x)) for x in line[-5:]]

                    lat, lon, alt, roll, pitch, yaw = line[:6]

                    if scale is None:
                        scale = np.cos(lat * np.pi / 180.)

                    R, t = pose_from_oxts_packet(lat, lon, alt, roll, pitch, yaw, scale)

                    if origin is None:
                        origin = t

                    R = R.reshape(3, 3)
                    t = t - origin

                    R = R @ self.calib['Tr_imu_velo'][:3, :3]
                    t = t + self.calib['Tr_imu_velo'][:3, 3]

                    Rs.append(R)
                    Ts.append(t)
        return Rs, Ts


    def load_calib(self):
        '''
        读取标定文件
        :return:
        '''
        calib_ = {}
        calib = {}
        with self.remote.get(self.calib_path) as calib_path:
            with open(calib_path, "r") as f:
                for line in f:
                    # 以第一个空格分割
                    key, value = line.strip().split(" ", 1)
                    value = np.array([float(x) for x in value.split()])
                    calib_[key] = value
        calib["P0"] = calib_["P0:"].reshape(3, 4)  # 投影矩阵
        calib["P1"] = calib_["P1:"].reshape(3, 4)
        calib["P2"] = calib_["P2:"].reshape(3, 4)
        calib["P2_inv"] = Tools.Math.inverse_rigid_trans(calib["P2"])
        calib["P3"] = calib_["P3:"].reshape(3, 4)
        calib["R0"] = calib_["R_rect"].reshape(3, 3)
        calib["R0_inv"] = np.linalg.inv(calib["R0"])
        calib["Tr_velo_cam"] = calib_["Tr_velo_cam"].reshape(3, 4)
        calib["Tr_imu_velo"] = calib_["Tr_imu_velo"].reshape(3, 4)
        calib["Tr_cam_velo"] = Tools.Math.inverse_rigid_trans(calib["Tr_velo_cam"])
        return calib


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


def pose_from_oxts_packet(lat, lon, alt, roll, pitch, yaw, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * lon * np.pi * er / 180.
    ty = scale * er * np.log(np.tan((90. + lat) * np.pi / 360.))
    tz = alt
    t = np.array([tx, ty, tz])
    Rx = rotx(roll)
    Ry = roty(pitch)
    Rz = rotz(yaw)
    R = Rz.dot(Ry.dot(Rx))
    return R, t


def rotx(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def load_image_to_memory(img_path):
    with open(img_path, 'rb') as file:
        img_data = file.read()
    image = Image.open(io.BytesIO(img_data))
    return image