# -*- coding: utf-8 -*-
from core import load_config, SceneLoader, Visualizer
import numpy as np

def filter(frame_data):
    # 绘制id为98的点云为灰色，其他点云为橙色
    frame_data.pcd.colors = np.ones_like(frame_data.pcd.points) * [1, 0.5, 0]  # 默认橙色
    frame_data.pcd.colors[frame_data.pcd.id == 98] = [0.5, 0.5, 0.5]  # 灰色
    return frame_data


if __name__ == '__main__':
    config = load_config('core/default_config.yaml')
    config.scene_config.dataset = 'carla4d'
    config.scene_config.scene_id = 0

    scene = SceneLoader(config.scene_config)
    visualizer = Visualizer(config.visualizer_config)

    # 对比两个帧（第0帧和第5帧）的点云，两个点云都使用体素化形式可视化
    config.visualizer_config.first_window.form = "voxel"
    config.visualizer_config.second_window.form = "voxel"
    visualizer.reset_config(config.visualizer_config)
    frame_data1 = scene.get_frame(frame_id=0, filter=filter)
    frame_data2 = scene.get_frame(frame_id=5, filter=filter)
    visualizer.compare_two_point_clouds(frame_data1, frame_data2)


    # 可视化第100帧的点云，使用体素化和八叉树形式对比
    config.visualizer_config.first_window.form = "voxel"
    config.visualizer_config.second_window.form = "octree"
    visualizer.reset_config(config.visualizer_config)
    visualizer.compare_one_frame(scene, frame_id=100, filter1=filter, filter2=filter)


    # 播放场景，使用体素化和八叉树形式对比
    config.visualizer_config.first_window.form = "point"
    config.visualizer_config.second_window.form = "octree"
    visualizer.reset_config(config.visualizer_config)
    visualizer.compare_scene(scene, filter1=filter, filter2=filter)