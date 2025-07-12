# -*- coding: utf-8 -*-
import yaml
from easydict import EasyDict
import argparse
import json

def load_config(config_path: str) -> EasyDict:
    """
    加载配置文件并返回一个 EasyDict 对象。
    :param config_path: 配置文件的路径
    :return: EasyDict 对象，包含配置参数
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    args = parse_args()
    # 如果命令行参数不为None，则覆盖配置文件中的相应参数
    config = merge_args_with_config(config, args)
    return EasyDict(config)



def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    :return: argparse.Namespace 对象，包含解析后的参数。
    """
    parser = argparse.ArgumentParser(description='config')
    # 场景加载参数
    parser.add_argument('--scene_config.datasets_yaml', type=str, default=None, help='数据集配置文件路径')
    parser.add_argument('--scene_config.dataset', type=str, default=None, help='数据集名称')
    parser.add_argument('--scene_config.scene_id', type=int, default=None, help='场景编号, 从0开始')

    # 窗口参数
    parser.add_argument('--visualizer_config.window_name', type=str, default=None, help='窗口名称')
    parser.add_argument('--visualizer_config.window_height', type=int, default=None, help='窗口高度')
    parser.add_argument('--visualizer_config.window_width', type=int, default=None, help='窗口宽度')
    parser.add_argument('--visualizer_config.window_left', type=int, default=None, help='窗口生成时的左边距')
    parser.add_argument('--visualizer_config.window_top', type=int, default=None, help='窗口生成时的上边距')

    # 渲染参数
    parser.add_argument('--visualizer_config.background_color', type=parse_list_config, default=None, help='背景颜色, RGB格式, 例如 [1.0, 1.0, 1.0] 表示白色, 单值范围为[0,1]')
    
    parser.add_argument('--visualizer_config.first_window.form', type=str, default=None, help='第一窗口的点云渲染形式, point表示点云，voxel表示体素， octree表示八叉树')
    parser.add_argument('--visualizer_config.first_window.point_size', type=float, default=None, help='第一个窗口点云的点大小')
    parser.add_argument('--visualizer_config.first_window.voxel_size', type=float, default=None, help='第一个窗口体素的大小')
    parser.add_argument('--visualizer_config.first_window.octree_max_depth', type=int, default=None, help='第一个窗口八叉树的最大深度')
    parser.add_argument('--visualizer_config.second_window.form', type=str, default=None, help='第二窗口的点云渲染形式, point表示点云，voxel表示体素， octree表示八叉树')
    parser.add_argument('--visualizer_config.second_window.point_size', type=float, default=None, help='第二个窗口点云的点大小')
    parser.add_argument('--visualizer_config.second_window.voxel_size', type=float, default=None, help='第二个窗口体素的大小')
    parser.add_argument('--visualizer_config.second_window.octree_max_depth', type=int, default=None, help='第二个窗口八叉树的最大深度')

    parser.add_argument('--visualizer_config.camera_sync', type=bool, default=None, help='是否同步两个窗口的相机位置')
    parser.add_argument('--visualizer_config.axis', type=float, default=None, help='坐标轴的长度, 等于0表示不显示坐标轴')
    parser.add_argument('--visualizer_config.init_camera_rpy', type=parse_list_config, default=None, help='初始相机的[roll, pitch, yaw]')
    parser.add_argument('--visualizer_config.init_camera_T', type=parse_list_config, default=None, help='初始相机位置[x, y, z]')

    args = parser.parse_args()
    return args


def merge_args_with_config(config: dict, args: argparse.Namespace) -> dict:
    """
    将命令行参数合并到配置字典中。
    :param config: 原始配置字典
    :param args: 命令行参数
    :return: 合并后的配置字典
    """
    for key, value in vars(args).items():
        if value is not None:
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
    return config

def parse_list_config(value: str) -> list:
    """
    解析配置中的列表或范围字符串。
    :param value: 配置字符串，可能是列表、范围或逗号分隔的数字
    :return: 解析后的列表
    """
    value = value.strip()
    try:
        # 尝试解析JSON格式
        return json.loads(value)
    except json.JSONDecodeError:
        # 尝试解析简单列表
        if value.startswith('[') and value.endswith(']'):
            stripped = value[1:-1].replace(' ', '')
            parts = [p for p in stripped.split(',') if p]
            return [int(p) for p in parts]
        # 尝试范围格式
        if '-' in value:
            start, end = value.split('-', 1)
            return [int(start), int(end)]
        # 尝试逗号分隔
        return [int(p) for p in value.split(',')]