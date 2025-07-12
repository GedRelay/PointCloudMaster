# -*- coding: utf-8 -*-
import os
import tqdm
import threading
import importlib
import paramiko
from contextlib import contextmanager
from easydict import EasyDict
import yaml


class SceneLoader:
    def __init__(self, config: EasyDict):
        self.datasets_yaml_file = config.datasets_yaml
        self.dataset_name = config.dataset
        self.scene_id = config.scene_id

        with open(self.datasets_yaml_file, 'r', encoding='utf-8') as f:
            datasets_config = EasyDict(yaml.safe_load(f))
        
        # 寻找datasets_config.datasets中name与self.dataset相同的配置
        dataset_config = None
        for dataset in datasets_config.datasets:
            if dataset.name == self.dataset_name:
                dataset_config = dataset
                break
        if dataset_config is None:
            raise ValueError(f"没有{self.dataset_name}这个数据集，请检查datasets.yaml文件")

        try:
            module = importlib.import_module(f'datasets.{dataset_config.name}')
        except ImportError as e:
            raise ImportError(f"无法导入数据集模块{dataset_config.name}，请检查datasets目录下是否存在{dataset_config.name}.py") from e
        try:
            dataset_class = getattr(module, dataset_config.name)
        except AttributeError:
            raise AttributeError(f"{dataset_config.name}.py中没有找到{dataset_config.name}类，请检查该模块是否正确实现") from None

        # 检查远程主机配置
        if dataset_config.get('hostname', None) is not None and dataset_config.hostname != "" and dataset_config.hostname != "localhost":
            # 在 datasets_config.hosts 中查找 hostname
            find_host = False
            for host in datasets_config.hosts:
                if host.hostname == dataset_config.hostname:
                    # 如果找到了对应的主机配置，则将其ip,username,private_key信息添加到dataset_config中
                    dataset_config.ip = host.ip
                    dataset_config.username = host.username
                    dataset_config.private_key = host.private_key
                    find_host = True
                    break
            if not find_host:
                raise ValueError(f"没有{dataset_config.hostname}这个主机，请检查datasets.yaml文件中的hosts配置")
        else:
            dataset_config.hostname = "localhost"
            dataset_config.ip = None
            dataset_config.username = None
            dataset_config.private_key = None
        
        self.dataset_loader = dataset_class(scene_id=self.scene_id, dataset_config=dataset_config)
        self.frame_num = self.dataset_loader.frame_num


    def get_frame(self, frame_id: int, filter=None) -> EasyDict:
        """
        加载某一帧的数据
        :param frame_id: 帧id
        :param filter: 过滤函数
        :return: frame_data: 当前帧的数据
        """
        assert 0 <= frame_id < self.frame_num, f'帧id:{frame_id}越界, 最大帧id为{self.frame_num - 1}'

        self.dataset_loader.init_frame_data()
        frame_data = self.dataset_loader.load_frame(frame_id)

        if filter is not None:
            frame_data = filter(frame_data)

        return frame_data


class RemoteClient:
    def __init__(self, hostname: str, ip: str, username: str, private_key: str):
        '''
        初始化远程客户端
        :param hostname: 远程主机名
        :param ip: 远程主机IP地址
        :param username: 远程主机用户名
        :param private_key: 远程主机私钥文件路径
        '''
        self.hostname = hostname
        self.is_remote = False if hostname == "" or hostname == "localhost" else True
        self.fetch_cache_path = "fetch_cache"
        if self.is_remote:
            os.makedirs(self.fetch_cache_path, exist_ok=True)
            # 连接远程服务器
            print(f'正在连接远程服务器{self.hostname}')
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                private_key = paramiko.RSAKey.from_private_key_file(private_key)
            except FileNotFoundError:
                raise FileNotFoundError(f"私钥文件{private_key}不存在，请检查datasets/datasets.yaml中的设置是否正确")
            self.ssh.connect(ip, port=22, username=username, pkey=private_key)
            self.sftp = self.ssh.open_sftp()

    def __del__(self):
        '''
        关闭远程连接
        '''
        if self.is_remote:
            if hasattr(self, 'sftp') and self.sftp:
                self.sftp.close()
            self.ssh.close()
            print(f'已关闭远程服务器{self.hostname}连接')

    @contextmanager
    def get(self, path: str):
        '''
        获取远程文件或本地文件的上下文管理器
        :param path: 文件路径
        :return: 本地文件路径
        '''
        # 下载文件到本地，并返回本地路径
        if self.is_remote:
            local_path = os.path.join(self.fetch_cache_path, os.path.basename(path))
            self.sftp.get(self.win2unix(path), local_path)
            yield local_path
            os.remove(local_path)
        else:
            yield path

    def listdir(self, path: str) -> list:
        '''
        列出远程目录或本地目录下的文件
        :param path: 目录路径
        :return: 文件列表
        '''
        if self.is_remote:
            return self.sftp.listdir(self.win2unix(path))
        else:
            return os.listdir(path)

    def win2unix(self, path: str) -> str:
        '''
        将Windows路径转换为Unix路径
        :param path: Windows路径
        :return: Unix路径
        '''
        return path.replace('\\', '/') if self.is_remote else path


class DatasetBase:
    def __init__(self, scene_id: int, dataset_config: EasyDict):
        '''
        初始化数据集加载器
        :param scene_id: 场景ID
        :param dataset_config: 数据集配置
        '''
        self.remote = RemoteClient(dataset_config.hostname, dataset_config.ip, dataset_config.username, dataset_config.private_key)
        self.root_path = dataset_config.root_path

        scene_config = None
        for scene in dataset_config.scenes:
            if scene.scene_id == scene_id:
                scene_config = scene
                break
        if scene_config is None:
            raise ValueError(f"没有{scene_id}这个场景，请检查datasets.yaml文件中的scenes配置")
        
        for key, value in scene_config.items():
            # 如果key以'path'结尾，则将其转换为绝对路径
            if isinstance(value, str) and key.endswith('path'):
                value = os.path.join(self.root_path, value)
            setattr(self, key, value)

        self.frame_num = len(self.remote.listdir(self.pcd_path))

        self.frame_data = None
        self.init_frame_data()

    def init_frame_data(self):
        self.frame_data = EasyDict({
            'pcd': {
                'points': None,  # 点云数据
                'colors': None,  # 点云颜色数据
            },
            'geometry': {
                'arrows': [],  # 箭头
                'spheres': [],  # 球体
                'boxes': [],  # 三维包围盒
            },
            'pose': {
                'R': None,  # 旋转矩阵
                'T': None,  # 平移向量
            }
        })

    def load_frame(self, frame_id: int) -> EasyDict:
        '''
        加载某一帧的数据
        :param frame_id: 帧id
        :return: frame_data: 当前帧的数据
        '''
        raise NotImplementedError('必须实现load_frame方法')

