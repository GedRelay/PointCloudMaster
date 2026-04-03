# -*- coding: utf-8 -*-
from core import load_config, SceneLoader, Visualizer, FrameData, Tools
import numpy as np
import cv2


# 运行方式： python -m demos.demo_dataset_aevascenes

def aevascenes_filter(frame_data: FrameData) -> FrameData:
    '''
    过滤函数：可视化3D包围盒和速度箭头
    :param frame_data: 帧数据
    '''
    print(frame_data)  # 可以打印帧数据查看其内容

    objects_num = len(frame_data.bbox_3d_corners)
    for i in range(objects_num):
        # 根据类别设置不同颜色
        cls = frame_data.class_names[i]
        if cls in ['car', 'bus', 'truck', 'trailer', 'vehicle_on_rails', 'other_vehicle']:
            box_color = (1, 0, 0)
            arrow_color = (1, 0, 0)
        elif cls in ['pedestrian', 'motorcyclist', 'bicyclist']:
            box_color = (0, 1, 0)
            arrow_color = (0, 1, 0)
        elif cls in ['bicycle', 'motorcycle', 'animal', 'traffic_item', 'traffic_sign']:
            box_color = (0, 0, 1)
            arrow_color = (0, 0, 1)
        elif cls in ['pole_trunk', 'building', 'other_structure', 'vegetation']:
            box_color = (1, 0, 1)
            arrow_color = (1, 0, 1)
        elif cls in ['road', 'lane_boundary', 'road_marking', 'reflective_markers', 'sidewalk', 'other_ground']:
            box_color = (0, 1, 1)
            arrow_color = (0, 1, 1)
        else:
            box_color = (1, 1, 1)
            arrow_color = (1, 1, 1)

        # 可视化速度箭头
        vector = np.array([frame_data.velocity_arrows_data[i][3], frame_data.velocity_arrows_data[i][4], frame_data.velocity_arrows_data[i][5]])
        start = np.array([frame_data.velocity_arrows_data[i][0], frame_data.velocity_arrows_data[i][1], frame_data.velocity_arrows_data[i][2]])
        frame_data.geometry.arrows.append(Tools.Geometry.get_arrow(vector, start, color=arrow_color))

        # 可视化3D包围盒
        frame_data.geometry.boxes.append(Tools.Geometry.get_box_by_corners(frame_data.bbox_3d_corners[i], color=box_color))

    frame_data.pcd.colors = np.zeros_like(frame_data.pcd.points) + 0.5  # 设置点云颜色为灰色

    # 可视化front_narrow_camera摄像头的图像
    WIDTH = frame_data.images['front_narrow_camera'].shape[1] // 4
    HEIGHT = frame_data.images['front_narrow_camera'].shape[0] // 4
    img = cv2.resize(frame_data.images['front_narrow_camera'], (WIDTH, HEIGHT))
    cv2.imshow('front_narrow_camera', img)

    return frame_data


if __name__ == '__main__':
    config = load_config('core/default_config.yaml')
    config.scene_config.dataset = 'aevascenes'
    config.scene_config.scene_id = 60

    # 加载场景
    scene = SceneLoader(config.scene_config)
    print("场景帧数:", scene.frame_num)

    # 创建可视化工具
    visualizer = Visualizer(config.visualizer_config)

    # 使用过滤函数动态可视化整个场景
    # 可以使用 空格键 暂停/继续，在暂停状态下可以使用方向键 ← → 或 ↑ ↓ 来控制帧的前进和后退
    visualizer.play_scene(scene, filter=aevascenes_filter)
