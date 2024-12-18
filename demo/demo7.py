# -*- coding: utf-8 -*-
"""
@Time        :  2024/12/17 15:42
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  demo7.py kitti_tracking数据集可视化
"""
import sys
sys.path.append('.')
from options import options  # 导入参数设置
from sceneloader import SceneLoader  # 导入场景加载器
from utils import Visualizer, Tools  # 导入可视化工具
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

def filter(pcd_xyz, other_data):
    other_data['pointinfo-color'] = np.ones_like(pcd_xyz)
    # 可视化3D包围盒
    other_data['geometry-bboxes'] = []
    for i, box in enumerate(other_data['bbox_corners_3d']):
        id = other_data['bbox_3d_ids'][i]
        if id not in bboxes_color:
            bboxes_color[id] = np.random.rand(3)
        bbox = Tools.get_bbox_by_corners(box, color=bboxes_color[id])
        other_data['geometry-bboxes'].append(bbox)

    # 筛选出指定区域的点云
    mask = (pcd_xyz[:, 0] > 10) & (pcd_xyz[:, 0] < 20) & (pcd_xyz[:, 1] > -8.5) & (pcd_xyz[:, 1] < -4.5)
    select_points = pcd_xyz[mask]
    other_data['pointinfo-color'][mask] = [1, 0, 0]

    # 可视化图片以及2D包围盒
    plt.imshow(other_data['image'])
    for i, box in enumerate(other_data['bbox_2d']):
        id = other_data['bbox_2d_ids'][i]
        plt.gca().add_patch(
            plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor=bboxes_color[id]))

    # 将筛选出的点云投影到图像上
    points_2d = points_to_image(select_points, other_data['calib']['Tr_velo_cam'], other_data['calib']['R0'],
                                other_data['calib']['P2'])
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c='r', s=1)
    plt.show()

    return pcd_xyz, other_data


if __name__ == '__main__':
    # 设置参数, 也可以在命令行中设置或者使用options.py的默认参数
    opt = options()
    opt.dataset = 'kitti_tracking'
    # opt.dataset = 'carla4d_dogshit'
    opt.scene_id = 0
    opt.preload = True  # 预加载
    opt.preload_begin = 0
    opt.preload_end = -1

    # 加载场景
    scene = SceneLoader(opt)
    print("场景帧数:", scene.frame_num)

    # 创建可视化工具
    visualizer = Visualizer(opt)

    # 动态可视化整个场景
    # 可以使用 空格键 暂停/继续，在暂停状态下可以使用方向键 ← → 或 ↑ ↓ 来控制帧的前进和后退
    visualizer.play_scene(scene, filter=filter, begin=0, end=-1, delay_time=0)
