# -*- coding: utf-8 -*-
"""
@Time        :  2024/6/4 23:39
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  demo1_filter1 演示1, 通过场景加载器加载场景并进行可视化, 使用过滤函数可视化过滤后的点云
"""
import sys
sys.path.append('.')
from options import options  # 导入参数设置
from sceneloader import SceneLoader  # 导入场景加载器
from utils import Visualizer  # 导入可视化工具

def filter(pcd_xyz, other_data):
    '''
    过滤函数, 去除id为98的点
    :param pcd_xyz: 点云
    :param other_data: 其他数据
    :return: pcd_xyz, other_data
    '''
    ids = other_data['pointinfo-id']
    # 去除id为98的点
    mask = ids != 98

    for key in other_data.keys():
        if key.startswith('pointinfo-'):
            other_data[key] = other_data[key][mask]
    pcd_xyz = pcd_xyz[mask]

    return pcd_xyz, other_data


if __name__ == '__main__':
    opt = options()
    opt.dataset = 'carla_4d'
    opt.scene_id = 0
    visualizer = Visualizer(opt)

    # 加载场景
    scene = SceneLoader(opt)
    print(scene.frame_num)

    # 获取某一帧的数据，并绘制点云
    pcd_xyz, _ = scene.get_frame(frame_id=100)
    print(pcd_xyz.shape)
    visualizer.draw_points(pcd_xyz)  # 绘制点云

    # visualizer.draw_one_frame(scene, frame_id=100)  # 绘制一帧

    # 使用过滤函数获取过滤后的点云，并绘制
    pcd_xyz_filter, _ = scene.get_frame(frame_id=100, filter=filter)
    print(pcd_xyz_filter.shape)
    visualizer.draw_points(pcd_xyz_filter)

    # visualizer.draw_one_frame(scene, frame_id=100, filter=filter)  # 绘制一帧，使用过滤函数

    # 播放场景
    visualizer.play_scene(scene, filter=filter)
