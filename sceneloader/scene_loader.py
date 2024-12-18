# -*- coding: utf-8 -*-
"""
@Time        :  2024/6/4 18:13
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  sceneloader  场景加载器
"""
import os
import tqdm
import threading
import json
import importlib


class SceneLoader:
    def __init__(self, opt):
        self.opt = opt

        json_path = os.path.join(os.path.dirname(__file__), 'datasets.json')
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        dataset_idx = -1
        for i, dataset in enumerate(json_data['datasets']):
            if dataset['name'] == self.opt.dataset:
                dataset_idx = i
                break
        if dataset_idx == -1:
            raise ValueError(f"没有{self.opt.dataset}这个数据集")

        dataset_loader_class_name = self.opt.dataset
        module = importlib.import_module(f'sceneloader.{self.opt.dataset}')
        dataset_loader_class = getattr(module, dataset_loader_class_name)

        if dataset_loader_class is None:
            raise ValueError(f"没有{dataset_loader_class_name}这个类")
        else:
            self.dataset_loader = dataset_loader_class(scene_id=self.opt.scene_id, json_data=json_data['datasets'][dataset_idx])

        self.frame_num = len(self.dataset_loader.filenames)

        self.__Rs = None
        self.__Ts = None
        self.__vehicle_state = None

        if self.opt.preload_end == -1:
            self.opt.preload_end = self.frame_num - 1
        self.preload_pcd_xyz_dict = {}
        self.preload_other_data_dict = {}
        if self.opt.preload:
            self.__preload_data()

    def __preload_data(self):
        """
        预加载数据
        :return:
        """
        assert 0 <= self.opt.preload_begin < self.frame_num, f'预加载起始帧:{self.opt.preload_begin}越界, 最大帧id为{self.frame_num - 1}'
        assert self.opt.preload_begin <= self.opt.preload_end < self.frame_num, f'预加载结束帧:{self.opt.preload_end}越界, 起始帧为{self.opt.preload_begin}, 最大帧id为{self.frame_num - 1}'

        # 多线程预加载
        with tqdm.tqdm(total=self.opt.preload_end - self.opt.preload_begin + 1, desc=f'数据预加载中（{self.opt.preload_begin}~{self.opt.preload_end}帧）', ncols=100) as bar:
            threads = []
            for frame_id in range(self.opt.preload_begin, self.opt.preload_end + 1):
                t = threading.Thread(target=self.__preload_data_thread, args=(frame_id, bar))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

    def __preload_data_thread(self, frame_id, bar):
        pcd_xyz, other_data = self.dataset_loader.load_frame(frame_id)
        self.preload_pcd_xyz_dict[frame_id] = pcd_xyz
        self.preload_other_data_dict[frame_id] = other_data
        bar.update(1)

    def get_frame(self, frame_id, filter=None):
        """
        加载某一帧的数据
        :param frame_id: 帧id
        :param filter: 过滤函数
        :return: pcd_xyz, 当前帧的点云数据
        :return: other_data, 当前帧的其他数据
        """
        assert 0 <= frame_id < self.frame_num, f'帧id:{frame_id}越界, 最大帧id为{self.frame_num - 1}'

        if self.opt.preload and frame_id in self.preload_pcd_xyz_dict:
            pcd_xyz = self.preload_pcd_xyz_dict[frame_id].copy()
            other_data = self.preload_other_data_dict[frame_id].copy()
        else:
            pcd_xyz, other_data = self.dataset_loader.load_frame(frame_id)

        # 添加位姿信息
        try:
            R, T = self.get_pose(frame_id)
            other_data['pose-R'] = R
            other_data['pose-T'] = T
        except:
            pass

        # 添加车辆状态信息
        try:
            if self.__vehicle_state is None:
                self.__vehicle_state = self.dataset_loader.load_vehicle_state(self.opt.scene_id)
            for key in self.__vehicle_state.keys():
                other_data[key] = self.__vehicle_state[key][frame_id]
        except:
            pass

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
        assert 0 <= frame_id < self.frame_num, f'帧id:{frame_id}越界, 最大帧id为{self.frame_num - 1}'

        if self.__Rs is None or self.__Ts is None:
            self.__Rs, self.__Ts = self.dataset_loader.load_poses(self.opt.scene_id)

        return self.__Rs[frame_id], self.__Ts[frame_id]


class DatasetLoader_Base:
    def __init__(self, scene_id, json_data):
        self.root_path = json_data['root_path']
        scenes = json_data['scenes']
        self.pcd_data_path = None
        self.pose_path = None
        self.vehicle_state_path = None
        for scene in scenes:
            if scene['scene_id'] == scene_id:
                self.pcd_data_path = os.path.join(self.root_path, scene['pcd_path'])
                if scene['pose_path'] is not None:
                    self.pose_path = os.path.join(self.root_path, scene['pose_path'])
                if scene['vehicle_state_path'] is not None:
                    self.vehicle_state_path = os.path.join(self.root_path, scene['vehicle_state_path'])
                break
        if self.pcd_data_path is None:
            raise ValueError(f'没有{scene_id}这个场景')

        self.filenames = os.listdir(self.pcd_data_path)
        try:
            self.filenames.sort(key=lambda x: int(x.split('.')[0]))  # 按照文件名排序
        except:
            pass

    def load_frame(self, frame_id):
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return: pcd_xyz, 当前帧的点云数据
        :return: other_data, 当前帧的其他数据
        '''
        raise NotImplementedError('必须实现load_frame方法')

    def load_poses(self, scene_id):
        '''
        获取所有帧的位姿
        :param scene_id: 场景id
        :return: Rs, 旋转矩阵列表 [N, 3, 3]
        :return: Ts, 平移向量列表 [N, 3]
        '''
        raise Exception('该数据集没有位姿信息')

    def load_vehicle_state(self, scene_id):
        '''
        获取所有帧的车辆状态
        :param scene_id: 场景id
        :return: vehicle_state, 字典，包含车辆状态信息
        '''
        raise Exception('该数据集没有车辆状态信息')

