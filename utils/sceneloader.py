# -*- coding: utf-8 -*-
"""
@Time        :  2024/6/4 18:13
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  sceneloader  场景加载器
"""
import os

import numpy as np

from utils.tools import Tools


class SceneLoader():
    def __init__(self, opt):
        self.opt = opt

        if self.opt.dataset == 'carla1':
            self.dataset_loader = DatasetLoader_Carla1(self.opt.scene_id)
        elif self.opt.dataset == 'carla2':
            self.dataset_loader = DatasetLoader_Carla2(self.opt.scene_id)
        elif self.opt.dataset == 'aeva':
            self.dataset_loader = DatasetLoader_Aeva(self.opt.scene_id)
        elif self.opt.dataset == 'helipr':
            self.dataset_loader = DatasetLoader_Helipr(self.opt.scene_id)

        self.frame_num = len(self.dataset_loader.filenames)

        self.__Rs = None
        self.__Ts = None

    def get_frame(self, frame_id, filter=None):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :param filter: 过滤函数
        :return: pcd_xyz, 当前帧的点云数据
        :return: other_data, 当前帧的其他数据
        '''
        assert 0 <= frame_id < self.frame_num, '帧id:{}越界, 最大帧id为{}'.format(frame_id, self.frame_num - 1)

        pcd_xyz, other_data = self.dataset_loader.load_frame(frame_id)

        if filter is not None:
            pcd_xyz, other_data = filter(pcd_xyz, other_data)

        return pcd_xyz, other_data

    def get_pose(self, frame_id):
        '''
        获取某一帧的位姿
        :param frame_id: 帧id
        :return: R, 旋转矩阵 [3, 3]
        :return: T, 平移向量 [3]
        '''
        assert 0 <= frame_id < self.frame_num, '帧id:{}越界, 最大帧id为{}'.format(frame_id, self.frame_num - 1)

        if self.__Rs is None or self.__Ts is None:
            self.__Rs, self.__Ts = self.dataset_loader.load_poses(self.opt.scene_id)

        return self.__Rs[frame_id], self.__Ts[frame_id]


class DatasetLoader_Base():
    def __init__(self, name, scene_id):
        self.name = name
        self.root_path = None
        self.pcd_data_path = None
        self.init_root_path()
        self.init_pcd_data_path(scene_id)

    def init_root_path(self):
        '''
        初始化数据集路径
        :return:
        '''
        raise NotImplementedError('必须实现该方法')

    def init_pcd_data_path(self, scene_id):
        '''
        初始化点云数据路径
        :param scene_id: 场景id
        :return:
        '''
        raise NotImplementedError('必顶实现该方法')

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return: pcd_xyz, 当前帧的点云数据
        :return: other_data, 当前帧的其他数据
        '''
        raise NotImplementedError('必须实现该方法')

    def load_poses(self, scene_id):
        '''
        获取所有帧的位姿
        :param scene_id: 场景id
        :return: Rs, 旋转矩阵列表 [N, 3, 3]
        :return: Ts, 平移向量列表 [N, 3]
        '''
        raise NotImplementedError('必须实现该方法')


class DatasetLoader_Carla1(DatasetLoader_Base):
    def __init__(self, scene_id):
        super(DatasetLoader_Carla1, self).__init__('carla1', scene_id)
        self.filenames = os.listdir(self.pcd_data_path)
        self.filenames.sort(key=lambda x: int(x.split('.')[0]))  # 按照文件名排序

    def init_root_path(self):
        '''
        初始化数据集路径
        :return:
        '''
        self.root_path = r'M:/datasetsss/4Dpcd/'

    def init_pcd_data_path(self, scene_id):
        '''
        初始化点云数据路径
        :param scene_id: 场景id
        :return:
        '''
        if scene_id == 0:
            self.pcd_data_path = self.root_path + 'Scene1/'
        elif scene_id == 1:
            self.pcd_data_path = self.root_path + 'Scene2/'
        else:
            raise ValueError('没有这个场景')

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return: pcd_xyz, 当前帧的点云数据
        :return: other_data, 当前帧的其他数据
        '''
        # x, y, z, rv, vx, vy, vz, id, label, intensity
        data = np.array(np.load(self.pcd_data_path + self.filenames[frame_id]).tolist())
        pcd_xyz = data[:, :3]
        other_data = {}
        other_data['pointinfo-rv'] = data[:, 3]
        other_data['pointinfo-real_v'] = data[:, 4:7]
        other_data['pointinfo-id'] = data[:, 7]
        other_data['pointinfo-intensity'] = data[:, 9]

        return pcd_xyz, other_data

    def load_poses(self, scene_id):
        '''
        获取所有帧的位姿
        :param scene_id: 场景id
        :return: Rs, 旋转矩阵列表 [N, 3, 3]
        :return: Ts, 平移向量列表 [N, 3]
        '''
        raise Exception('该数据集没有位姿信息')


class DatasetLoader_Carla2(DatasetLoader_Base):
    def __init__(self, scene_id):
        super(DatasetLoader_Carla2, self).__init__('carla2', scene_id)
        self.filenames = os.listdir(self.pcd_data_path)
        self.filenames.sort(key=lambda x: int(x.split('.')[0]))  # 按照文件名排序

    def init_root_path(self):
        '''
        初始化数据集路径
        :return:
        '''
        self.root_path = r'M:/datasetsss/4Dcarla_compensate_demo/'

    def init_pcd_data_path(self, scene_id):
        '''
        初始化点云数据路径
        :param scene_id: 场景id
        :return:
        '''
        if scene_id == 0:
            self.pcd_data_path = self.root_path + 'carla_compensate_demo/'
        else:
            raise ValueError('没有这个场景')

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return: pcd_xyz, 当前帧的点云数据
        :return: other_data, 当前帧的其他数据
        '''
        # x, y, z, cosangle, rv, vx, vy, vz, vcps, id, label, intensity
        data = np.array(np.load(self.pcd_data_path + self.filenames[frame_id]).tolist())
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
        with open(self.root_path + 'poses.csv', 'r') as f:
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


class DatasetLoader_Aeva(DatasetLoader_Base):
    def __init__(self, scene_id):
        super(DatasetLoader_Aeva, self).__init__('aeva', scene_id)
        self.filenames = os.listdir(self.pcd_data_path)
        self.filenames.sort(key=lambda x: int(x.split('.')[0]))  # 按照文件名排序

    def init_root_path(self):
        '''
        初始化数据集路径
        :return:
        '''
        self.root_path = r'M:/datasetsss/aeva/'

    def init_pcd_data_path(self, scene_id):
        '''
        初始化点云数据路径
        :param scene_id: 场景id
        :return:
        '''
        if scene_id == 0:
            self.pcd_data_path = self.root_path + '00/frames/'
        elif scene_id == 1:
            self.pcd_data_path = self.root_path + '01/frames/'
        elif scene_id == 2:
            self.pcd_data_path = self.root_path + '02/frames/'
        elif scene_id == 3:
            self.pcd_data_path = self.root_path + '03/frames/'
        elif scene_id == 4:
            self.pcd_data_path = self.root_path + '04/frames/'
        elif scene_id == 5:
            self.pcd_data_path = self.root_path + '05/frames/'
        elif scene_id == 6:
            self.pcd_data_path = self.root_path + '06/frames/'
        elif scene_id == 7:
            self.pcd_data_path = self.root_path + '07/frames/'
        else:
            raise ValueError('没有这个场景')

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return:
        '''
        # x, y, z, rv, time
        data = np.fromfile(self.pcd_data_path + self.filenames[frame_id], dtype=np.float32).reshape(-1, 5)
        pcd_xyz = data[:, :3]
        other_data = {}
        other_data['pointinfo-rv'] = data[:, 3]
        other_data['pointinfo-time'] = data[:, 4]

        return pcd_xyz, other_data

    def load_poses(self, scene_id):
        '''
        获取所有帧的位姿
        :param scene_id: 场景id
        :return: Rs, 旋转矩阵列表 [N, 3, 3]
        :return: Ts, 平移向量列表 [N, 3]
        '''
        if scene_id == 0:
            pose_txt = self.root_path + '00/aeva_poses.txt'
        elif scene_id == 1:
            pose_txt = self.root_path + '01/aeva_poses.txt'
        elif scene_id == 2:
            pose_txt = self.root_path + '02/aeva_poses.txt'
        elif scene_id == 3:
            pose_txt = self.root_path + '03/aeva_poses.txt'
        elif scene_id == 4:
            pose_txt = self.root_path + '04/aeva_poses.txt'
        elif scene_id == 5:
            pose_txt = self.root_path + '05/aeva_poses.txt'
        elif scene_id == 6:
            pose_txt = self.root_path + '06/aeva_poses.txt'
        elif scene_id == 7:
            pose_txt = self.root_path + '07/aeva_poses.txt'
        else:
            raise ValueError('没有这个场景')

        poses = np.loadtxt(pose_txt).reshape(-1, 3, 4)

        Rs = poses[:, :3, :3]
        Ts = poses[:, :3, 3]

        return Rs, Ts


class DatasetLoader_Helipr(DatasetLoader_Base):
    def __init__(self, scene_id):
        super(DatasetLoader_Helipr, self).__init__('helipr', scene_id)
        self.filenames = os.listdir(self.pcd_data_path)
        self.filenames.sort(key=lambda x: int(x.split('.')[0]))  # 按照文件名排序

    def init_root_path(self):
        '''
        初始化数据集路径
        :return:
        '''
        self.root_path = r'M:/datasetsss/HeLiPR/'

    def init_pcd_data_path(self, scene_id):
        '''
        初始化点云数据路径
        :param scene_id: 场景id
        :return:
        '''
        if scene_id == 0:
            self.pcd_data_path = self.root_path + 'KAIST05/LiDAR/Aeva/'
        else:
            raise ValueError('没有这个场景')

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return:
        '''
        # x, y, z, reflectivity, vcps, time, line_index, intensity
        dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('reflectivity', np.float32),
                          ('velocity', np.float32), ('time_offset_ns', np.int32), ('line_index', np.uint8),
                          ('intensity', np.float32)])
        data = np.array(np.fromfile(self.pcd_data_path + self.filenames[frame_id], dtype=dtype).tolist())
        pcd_xyz = data[:, :3]
        other_data = {}
        other_data['pointinfo-vcps'] = data[:, 4]
        other_data['pointinfo-intensity'] = data[:, 7]
        other_data['pointinfo-time'] = data[:, 5]
        other_data['pointinfo-line_index'] = data[:, 6]
        other_data['pointinfo-reflectivity'] = data[:, 3]

        return pcd_xyz, other_data

    def load_poses(self, scene_id):
        '''
        获取所有帧的位姿
        :param scene_id: 场景id
        :return: Rs, 旋转矩阵列表 [N, 3, 3]
        :return: Ts, 平移向量列表 [N, 3]
        '''
        raise Exception('该数据集没有位姿信息')
