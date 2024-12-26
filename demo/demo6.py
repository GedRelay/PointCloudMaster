# -*- coding: utf-8 -*-
"""
@Time        :  2024/9/6 20:02
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  demo6 演示6, 体素化与八叉树形式可视化演示
"""
import sys
sys.path.append('.')
from options import options  # 导入参数设置
from sceneloader import SceneLoader  # 导入场景加载器
from utils import Visualizer

import numpy as np

def filter(pcd_xyz, other_data):
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


if __name__ == '__main__':
    opt = options()
    opt.window_left = 0
    opt.dataset = 'carla_4d'
    opt.scene_id = 0

    opt.preload = True # 预加载
    opt.preload_end = 100 # 加载第0到第100帧

    # 可视化器
    visualizer = Visualizer(opt)

    # 加载场景
    scene = SceneLoader(opt)

    # 对比两个帧（第0帧和第5帧）的点云，两个点云都使用体素化形式可视化
    pcd_xyz1, other_data1 = scene.get_frame(frame_id=0, filter=filter)
    pcd_xyz2, other_data2 = scene.get_frame(frame_id=5, filter=filter)
    visualizer.compare_two_point_clouds(pcd_xyz1, pcd_xyz2, other_data1, other_data2,
                                        form1="voxel", voxel_size1=0.5,
                                        form2="voxel", voxel_size2=0.5)

    # 可视化第100帧的点云，使用体素化和八叉树形式对比
    visualizer.compare_one_frame(scene, frame_id=100, filter1=filter, filter2=filter,
                                 form1="voxel", voxel_size1=0.5,
                                 form2="octree", octree_max_depth2=8)

    # 播放场景，使用体素化和八叉树形式对比
    # 可以使用 空格键 暂停/继续，在暂停状态下可以使用方向键 ← → 或 ↑ ↓ 来控制帧的前进和后退
    visualizer.compare_scene(scene, filter1=filter, filter2=filter, delay_time=0, begin=0, end=-1,
                                form1="point", point_size1=2,
                                form2="octree", octree_max_depth2=8,
                                axis=5, init_camera_rpy=[180, -60, -90], init_camera_T=[15, 0, 30])