# -*- coding: utf-8 -*-
from core import load_config, SceneLoader, Visualizer, Filters
import numpy as np


def filter1(frame_data):
    # 绘制id为98的点云为灰色，其他点云为橙色
    frame_data.pcd.colors = np.ones_like(frame_data.pcd.points) * [1, 0.5, 0]  # 默认橙色
    frame_data.pcd.colors[frame_data.pcd.id == 98] = [0.5, 0.5, 0.5]  # 灰色

    return frame_data

def filter2(frame_data):
    # 随机删除90%的点云
    point_num = frame_data.pcd.points.shape[0]
    left_num = int(point_num * 0.1)
    mask = np.random.choice(point_num, left_num, replace=False)
    frame_data = Filters.remain_points_by_mask(frame_data, mask)

    return frame_data


if __name__ == '__main__':
    config = load_config('core/default_config.yaml')
    config.scene_config.dataset = 'carla4d'
    config.scene_config.scene_id = 0

    scene = SceneLoader(config.scene_config)
    visualizer = Visualizer(config.visualizer_config)
    
    # 两个不同帧的点云使用同一个过滤函数，进行对比
    frame_data1 = scene.get_frame(frame_id=0, filter=filter1)
    frame_data2 = scene.get_frame(frame_id=5, filter=filter1)
    visualizer.compare_two_point_clouds(frame_data1, frame_data2)

    # 使用两个不同的过滤函数对比同一帧的点云
    visualizer.compare_one_frame(scene, frame_id=100, filter1=filter1, filter2=filter2)

    # 使用两个过滤函数对比动态场景的点云
    visualizer.compare_scene(scene, filter1=filter1, filter2=filter2, frame_range=[0, -1], delay_time=0)