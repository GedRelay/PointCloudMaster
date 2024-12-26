# -*- coding: utf-8 -*-
"""
@Time        :  2024/6/5 上午12:03
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  demo2 演示2, id为98的点云为灰色，其他点云为橙色，此外为每个id绘制候选框
"""
import sys
sys.path.append('.')
from options import options
from sceneloader import SceneLoader  # 导入场景加载器
from utils import Visualizer, Tools  # 工具类，使用其中的get_bbox_from_points函数从点云中获取包围盒
import numpy as np

def filter(pcd_xyz, other_data):
    '''
    过滤函数, id为98的点云为灰色，其他点云为橙色，此外为每个id绘制候选框
    :param pcd_xyz: 点云
    :param other_data: 其他数据
    :return: pcd_xyz, other_data
    '''

    # 绘制id为98的点云为灰色，其他点云为橙色
    color = np.ones_like(pcd_xyz) * [1, 0.5, 0]
    color[other_data['pointinfo-id'] == 98] = [0.5, 0.5, 0.5]
    other_data['pointinfo-color'] = color

    # 绘制包围盒
    other_data['geometry-bboxes'] = []
    for id in np.unique(other_data['pointinfo-id']):
        if id == 98:
            continue
        mask = other_data['pointinfo-id'] == id
        points = pcd_xyz[mask]
        if points.shape[0] >= 4:
            bbox = Tools.get_bbox_from_points(points, color=(0, 1, 0))
            other_data['geometry-bboxes'].append(bbox)

    return pcd_xyz, other_data


if __name__ == '__main__':
    opt = options()
    opt.dataset = 'carla_4d'
    opt.scene_id = 0
    visualizer = Visualizer(opt)

    # 加载场景
    scene = SceneLoader(opt)
    print(scene.frame_num)

    # 使用过滤函数绘制点云
    visualizer.draw_one_frame(scene, frame_id=100, filter=filter)

    # 播放场景，使用过滤函数
    visualizer.play_scene(scene, filter=filter)
