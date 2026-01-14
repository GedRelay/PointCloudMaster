# -*- coding: utf-8 -*-
from core import load_config, SceneLoader, Visualizer, Tools
import matplotlib.pyplot as plt
import numpy as np


def points_to_image(points, Tr_velo_cam, R0, P):
    """
    将3D点投影到2D图像上
    :param points: 3D点 (N, 3)
    :param Tr_velo_cam: 从雷达坐标系到相机坐标系的变换矩阵 (3, 4)
    :param R0: 从相机坐标系到图像坐标系的变换矩阵 (3, 3)
    :param P: 投影矩阵 (3, 4)
    :return: 2D点 (N, 2)
    """
    pts_3d_velo = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)
    pts_3d_ref = pts_3d_velo @ Tr_velo_cam.T  # (N, 3)
    pts_3d_rect = (R0 @ pts_3d_ref.T).T  # (N, 3)
    pts_3d_rect = np.hstack([pts_3d_rect, np.ones((pts_3d_rect.shape[0], 1))])  # (N, 4)
    pts_2d = pts_3d_rect @ P.T  # (N, 3)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

bboxes_color = {}  # 每个id拥有不同的颜色

def filter(frame_data):
    frame_data.pcd.colors = np.ones_like(frame_data.pcd.points)

    # 可视化3D包围盒
    for i, bbox_corner in enumerate(frame_data.bbox_3d_corners):
        id = frame_data.bbox_3d_ids[i]
        if id not in bboxes_color:
            bboxes_color[id] = np.random.rand(3)
        bbox = Tools.Geometry.get_box_by_corners(bbox_corner, color=bboxes_color[id])
        frame_data.geometry.boxes.append(bbox)

    # 筛选出指定区域的点云
    mask = (frame_data.pcd.points[:, 0] > 10) & (frame_data.pcd.points[:, 0] < 20) & (frame_data.pcd.points[:, 1] > -8.5) & (frame_data.pcd.points[:, 1] < -4.5)
    select_points = frame_data.pcd.points[mask]
    frame_data.pcd.colors[mask] = [1, 0, 0]

    # 可视化图片以及2D包围盒
    plt.clf()
    plt.imshow(frame_data.image)
    for i, box in enumerate(frame_data.bbox_2d):
        id = frame_data.bbox_2d_ids[i]
        plt.gca().add_patch(
            plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor=bboxes_color[id]))

    # 将筛选出的点云投影到图像上
    points_2d = points_to_image(select_points, frame_data.calib['Tr_velo_cam'], frame_data.calib['R0'],
                                frame_data.calib['P2'])
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c='r', s=1)
    plt.ion()
    plt.show()

    return frame_data


if __name__ == '__main__':
    config = load_config('core/default_config.yaml')
    config.scene_config.dataset = 'kitti_tracking'
    config.scene_config.scene_id = 0
    config.visualizer_config.background_color = [0, 0, 0]

    scene = SceneLoader(config.scene_config)
    visualizer = Visualizer(config.visualizer_config)

    visualizer.play_scene(scene, filter=filter)
