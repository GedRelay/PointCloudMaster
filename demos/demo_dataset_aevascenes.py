# -*- coding: utf-8 -*-
from core import load_config, SceneLoader, Visualizer, FrameData, Tools
import numpy as np
import cv2


# 运行方式： python -m demos.demo_dataset_aevascenes

def points_to_image(points, calib, camera_name):
    T_v_c = np.linalg.inv(calib['vehicle_to_camera_extrinsics'][camera_name])

    K = calib['camera_intrinsics'][camera_name]['intrinsic_matrix']
    dist = np.array(calib['camera_intrinsics'][camera_name]['distortion_coefficients'])

    N = points.shape[0]
    points_h = np.concatenate([points, np.ones((N, 1))], axis=1)

    points_camera = (T_v_c @ points_h.T).T[:, :3]

    mask = points_camera[:, 2] > 0
    xyz_valid = points_camera[mask]

    rvec = np.zeros(3)
    tvec = np.zeros(3)

    uv, _ = cv2.projectPoints(
        xyz_valid,
        rvec,
        tvec,
        K,
        dist
    )

    return uv.reshape(-1, 2)



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
            box_color = (0, 0, 0)
            arrow_color = (0, 0, 0)

        # 可视化速度箭头
        vector = np.array([frame_data.velocity_arrows_data[i][3], frame_data.velocity_arrows_data[i][4], frame_data.velocity_arrows_data[i][5]])
        start = np.array([frame_data.velocity_arrows_data[i][0], frame_data.velocity_arrows_data[i][1], frame_data.velocity_arrows_data[i][2]])
        frame_data.geometry.arrows.append(Tools.Geometry.get_arrow(vector, start, color=arrow_color))

        # 可视化3D包围盒
        frame_data.geometry.boxes.append(Tools.Geometry.get_box_by_corners(frame_data.bbox_3d_corners[i], color=box_color))

    frame_data.pcd.colors = np.zeros_like(frame_data.pcd.points) + 0.5  # 设置点云颜色为灰色

    # 筛选出指定区域的点云
    mask = (frame_data.pcd.points[:, 0] > 10) & (frame_data.pcd.points[:, 0] < 20) & (frame_data.pcd.points[:, 1] > -6.5) & (frame_data.pcd.points[:, 1] < -2.5)
    select_points = frame_data.pcd.points[mask]
    frame_data.pcd.colors[mask] = [1, 0, 0]

    # 将筛选出的点云投影到图像上
    image = frame_data.images['front_wide_camera']

    points_2d = points_to_image(select_points, frame_data.calib, 'front_wide_camera')
    for point in points_2d:
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

    # 可视化front_wide_camera摄像头的图像
    WIDTH = image.shape[1] // 4
    HEIGHT = image.shape[0] // 4
    image = cv2.resize(image, (WIDTH, HEIGHT))
    cv2.imshow('front_wide_camera', image)


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
