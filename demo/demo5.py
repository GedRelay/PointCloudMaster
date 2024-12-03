# -*- coding: utf-8 -*-
"""
@Time        :  2024/9/2 20:14
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  demo5.py 对比两个窗口的点云
"""
import sys
sys.path.append('.')
from options import options  # 导入参数设置
from sceneloader import SceneLoader  # 导入场景加载器
from utils import Visualizer, Filters

import numpy as np

def filter1(pcd_xyz, other_data):
    '''
    过滤函数1: 将id为98的点云设置为灰色
    :param pcd_xyz: 点云
    :param other_data: 其他数据
    :return: pcd_xyz, other_data
    '''
    # 绘制id为98的点云为灰色，其他点云为橙色
    color = np.ones_like(pcd_xyz) * [1, 0.5, 0]
    color[other_data['pointinfo-id'] == 98] = [0.5, 0.5, 0.5]
    other_data['pointinfo-color'] = color

    # print('filter1, point num:', pcd_xyz.shape[0])

    return pcd_xyz, other_data

def filter2(pcd_xyz, other_data):
    '''
    过滤函数2: 随机删除90%的点云
    :param pcd_xyz: 点云
    :param other_data: 其他数据
    :return: pcd_xyz, other_data
    '''
    # 随机删除90%的点云
    point_num = pcd_xyz.shape[0]
    left_num = int(point_num * 0.1)
    mask = np.random.choice(point_num, left_num, replace=False)
    pcd_xyz, other_data = Filters.remove_points_by_mask(pcd_xyz, other_data, mask)

    # print('filter2, point num:', pcd_xyz.shape[0])

    return pcd_xyz, other_data


if __name__ == '__main__':
    opt = options()
    opt.window_left = 0
    opt.dataset = 'carla1'
    opt.scene_id = 0

    # 可视化器
    visualizer = Visualizer(opt)

    # 加载场景
    scene = SceneLoader(opt)

    # 对比两个点云
    pcd_xyz1, other_data1 = scene.get_frame(frame_id=0, filter=filter1)
    pcd_xyz2, other_data2 = scene.get_frame(frame_id=5, filter=filter1)
    visualizer.compare_two_point_clouds(pcd_xyz1, pcd_xyz2, other_data1, other_data2)

    # 使用两个过滤函数对比单帧中的点云
    visualizer.compare_one_frame(scene, frame_id=100, filter1=filter1, filter2=filter2)

    # 使用两个过滤函数对比动态场景的点云
    # 可以使用 空格键 暂停/继续，在暂停状态下可以使用方向键 ← → 或 ↑ ↓ 来控制帧的前进和后退
    visualizer.compare_scene(scene, filter1=filter1, filter2=filter2, delay_time=0, begin=0, end=-1, axis=5,
                          init_camera_rpy=[180, -60, -90], init_camera_T=[15, 0, 30])