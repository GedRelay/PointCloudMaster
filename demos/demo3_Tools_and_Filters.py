# -*- coding: utf-8 -*-
from core import load_config, SceneLoader, Visualizer, Tools, Filters
import numpy as np

# Tools中包含了一些常用的工具函数，Filter中包含了一些常用的过滤函数

# 定义过滤函数
def work(frame_data):
    '''
    过滤函数：
    1. 去除id为98的点云数据
    2. 为每个id绘制三维框
    3. 显示物体的速度箭头
    4. 每个id的点云数据使用不同颜色绘制
    :param frame_data: 帧数据
    '''
    target_id = 98
    mask = (frame_data.pcd.id != target_id)
    frame_data = Filters.remain_points_by_mask(frame_data, mask)  # 使用Filters类中的方法来过滤点云数据

    unique_id = np.unique(frame_data.pcd.id)
    frame_data.pcd.colors = np.zeros((frame_data.pcd.points.shape[0], 3), dtype=np.float32)  # 初始化颜色数组
    for id in unique_id:
        points = frame_data.pcd.points[frame_data.pcd.id == id]
        # 根据点云获取三维框
        if points.shape[0] >= 4:  # 至少需要4个点才能构成一个三维框
            box = Tools.Geometry.get_box_from_points(points=points,
                                                    color=(1, 0, 0),
                                                    oriented=False)
            frame_data.geometry.boxes.append(box)  # 将三维框添加到帧数据中

        # 获取速度箭头
        velocity = frame_data.pcd.velocity[frame_data.pcd.id == id][0]
        if np.linalg.norm(velocity) > 1e-3:  # 如果速度接近零，则不绘制箭头
            arrow = Tools.Geometry.get_arrow(vector=velocity,
                                            start=points.mean(axis=0),
                                            color=(0, 1, 0))
            frame_data.geometry.arrows.append(arrow)  # 将速度箭头添加到帧数据中
        
        # 为每个id的点云数据分配随机颜色
        frame_data.pcd.colors[frame_data.pcd.id == id] = np.random.rand(3)  # 随机颜色

    return frame_data


if __name__ == '__main__':
    # 加载参数和设置
    config = load_config('core/default_config.yaml')
    config.scene_config.dataset = 'carla4d'
    config.scene_config.scene_id = 0
    visualizer = Visualizer(config.visualizer_config)

    # 加载场景
    scene = SceneLoader(config.scene_config)

    # 播放场景
    visualizer.play_scene(scene, filter=work)
