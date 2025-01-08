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
import paramiko
from contextlib import contextmanager


class SceneLoader:
    def __init__(self, opt=None, dataset='carla_4d', scene_id=0, preload=False, preload_begin=0, preload_end=-1, jsonfile='datasets.json'):
        # 注意如果opt不为None时后面的参数都不可用
        self.dataset = dataset
        self.scene_id = scene_id
        self.preload = preload
        self.preload_begin = preload_begin
        self.preload_end = preload_end
        self.jsonfile = jsonfile
        self.__load_settings_by_opt(opt)

        json_path = os.path.join(os.path.dirname(__file__), self.jsonfile)
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        dataset_idx = -1
        for i, dataset in enumerate(json_data['datasets']):
            if dataset['name'] == self.dataset:
                dataset_idx = i
                break
        if dataset_idx == -1:
            raise ValueError(f"没有{self.dataset}这个数据集")

        dataset_loader_class_name = self.dataset
        module = importlib.import_module(f'sceneloader.{self.dataset}')
        dataset_loader_class = getattr(module, dataset_loader_class_name)

        if dataset_loader_class is None:
            raise ValueError(f"没有{dataset_loader_class_name}这个类")
        else:
            self.dataset_loader = dataset_loader_class(scene_id=self.scene_id, json_data=json_data['datasets'][dataset_idx])

        self.frame_num = len(self.dataset_loader.filenames)

        if self.preload_end == -1:
            self.preload_end = self.frame_num - 1
        self.preload_pcd_xyz_dict = {}
        self.preload_other_data_dict = {}
        if self.preload:
            if self.dataset_loader.remote.is_remote:
                self.__preload_data_remote()
            else:
                self.__preload_data_local()

    def __load_settings_by_opt(self, opt):
        if opt is not None:
            self.dataset = opt.dataset
            self.scene_id = opt.scene_id
            self.preload = opt.preload
            self.preload_begin = opt.preload_begin
            self.preload_end = opt.preload_end
            self.jsonfile = opt.jsonfile

    def __preload_data_local(self):
        """
        预加载数据，本地加载，使用多线程
        :return:
        """
        assert 0 <= self.preload_begin < self.frame_num, f'预加载起始帧:{self.preload_begin}越界, 最大帧id为{self.frame_num - 1}'
        assert self.preload_begin <= self.preload_end < self.frame_num, f'预加载结束帧:{self.preload_end}越界, 起始帧为{self.preload_begin}, 最大帧id为{self.frame_num - 1}'

        # 多线程预加载
        with tqdm.tqdm(total=self.preload_end - self.preload_begin + 1, desc=f'数据预加载中（{self.preload_begin}~{self.preload_end}帧）', ncols=100) as bar:
            threads = []
            for frame_id in range(self.preload_begin, self.preload_end + 1):
                t = threading.Thread(target=self.__preload_data_thread, args=(frame_id, bar))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

    def __preload_data_remote(self):
        """
        预加载数据，远程加载，使用单线程
        :return:
        """
        assert 0 <= self.preload_begin < self.frame_num, f'预加载起始帧:{self.preload_begin}越界, 最大帧id为{self.frame_num - 1}'
        assert self.preload_begin <= self.preload_end < self.frame_num, f'预加载结束帧:{self.preload_end}越界, 起始帧为{self.preload_begin}, 最大帧id为{self.frame_num - 1}'

        # 单线程预加载
        with tqdm.tqdm(total=self.preload_end - self.preload_begin + 1, desc=f'数据预加载中（{self.preload_begin}~{self.preload_end}帧）', ncols=100) as bar:
            for frame_id in range(self.preload_begin, self.preload_end + 1):
                self.__preload_data_thread(frame_id, bar)

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

        if self.preload and frame_id in self.preload_pcd_xyz_dict:
            pcd_xyz = self.preload_pcd_xyz_dict[frame_id].copy()
            other_data = self.preload_other_data_dict[frame_id].copy()
        else:
            pcd_xyz, other_data = self.dataset_loader.load_frame(frame_id)

        if filter is not None:
            pcd_xyz, other_data = filter(pcd_xyz, other_data)

        return pcd_xyz, other_data


class RemoteClient:
    def __init__(self, hostname):
        self.hostname = hostname
        self.is_remote = False if hostname == "" or hostname == "localhost" else True
        self.local_tmp_path = "temp"
        remote_ip = None
        port = None
        username = None
        private_key = None
        os.makedirs(self.local_tmp_path, exist_ok=True)
        if self.is_remote:
            host_json_path = os.path.join(os.path.dirname(__file__), 'hosts.json')
            with open(host_json_path, 'r') as f:
                json_data = json.load(f)
                for host in json_data['hosts']:
                    if host['hostname'] == hostname:
                        remote_ip = host['ip']
                        username = host['username']
                        port = host['port'] if 'port' in host else 22
                        private_key = paramiko.RSAKey.from_private_key_file(host['private_key'])
                        break
            if remote_ip is None:
                raise ValueError(f'没有{hostname}这个主机, 请检查hosts.json文件')
            # 连接远程服务器
            print(f'连接远程服务器{self.hostname}')
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(remote_ip, port=port, username=username, pkey=private_key)
            self.sftp = self.ssh.open_sftp()

    def __del__(self):
        if self.is_remote:
            self.sftp.close()
            self.ssh.close()

    @contextmanager
    def get(self, path):
        # 下载文件到本地，并返回本地路径
        if self.is_remote:
            local_path = os.path.join(self.local_tmp_path, os.path.basename(path))
            self.sftp.get(self.win2unix(path), local_path)
            yield local_path
            os.remove(local_path)
        else:
            yield path

    def listdir(self, path):
        if self.is_remote:
            return self.sftp.listdir(self.win2unix(path))
        else:
            return os.listdir(path)

    def win2unix(self, path):
        if self.is_remote:
            return path.replace('\\', '/')
        else:
            return path

class DatasetLoader_Base:
    def __init__(self, scene_id, json_data):
        self.remote = RemoteClient(json_data['hostname'] if 'hostname' in json_data and json_data['hostname'] else "")
        self.root_path = json_data['root_path']
        scenes = json_data['scenes']
        self.pcd_data_path = None
        for scene in scenes:
            if scene['scene_id'] == scene_id:
                self.pcd_data_path = os.path.join(self.root_path, scene['pcd_path'])
                break
        if self.pcd_data_path is None:
            raise ValueError(f'没有{scene_id}这个场景')

        self.filenames = self.remote.listdir(self.pcd_data_path)
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

