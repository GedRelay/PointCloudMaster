# -*- coding: utf-8 -*-
"""
@Time        :  2024/8/29 16:57
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  test1.py
"""

from options import Options
from utils.visualizer import Visualizer
from utils.sceneloader import SceneLoader
from utils.filters import Filters
from utils.tools import Tools

import time
import numpy as np

# aeva
zmin = -1.5
zmax = 3
vbuffer = 1.5
epsilon = 1.5
min_samples = 15

# carla1
# zmin = -2.2
# zmax = 3
# vbuffer = 0.7
# epsilon = 1
# min_samples = 5

def work(pcd_xyz, other_data):

    other_data['pointinfo-id'] = np.arange(pcd_xyz.shape[0])
    pcd_xyz_copy = pcd_xyz.copy()
    other_data_copy = other_data.copy()

    pre_point_num = pcd_xyz.shape[0]

    v = other_data['pointinfo-rv']
    # v_real = v * sqrt(x^2 + y^2 + z^2) / x
    v = v * np.linalg.norm(pcd_xyz, axis=1) / pcd_xyz[:, 0]

    # 以从-50到50，以1为间隔的分布中，统计哪个区间的点数最多，得到该区间的中心值
    hist, bins = np.histogram(v, bins=np.arange(-50, 50, 1))
    center = bins[np.argmax(hist)] + 0.5

    # 选出v在[center-buffer, center+buffer]范围内的点
    mask = (v >= center - vbuffer) & (v <= center + vbuffer)

    # 移除mask内的点
    pcd_xyz, other_data = Filters.remove_points_by_mask(pcd_xyz, other_data, mask)

    # 移除z轴范围外的点
    pcd_xyz, other_data = Filters.remain_points_by_z_axis(pcd_xyz, other_data, z_min=zmin, z_max=zmax)

    print("点云数量：", pcd_xyz.shape[0], "原点云数量：", pre_point_num, "比例：", pcd_xyz.shape[0] / pre_point_num * 100)

    # 聚类
    # if pcd_xyz.shape[0] >= 5:
    #     labels = Tools.dbscan(pcd_xyz, eps=epsilon, min_samples=min_samples)
    #     labels_unique = np.unique(labels)
    #
    #     colors_list = np.random.rand(len(labels_unique), 3)
    #     colors = np.zeros((len(pcd_xyz), 3))
    #     for i, label in enumerate(labels_unique):
    #         colors[labels == label] = colors_list[i]
    #     other_data['pointinfo-color'] = colors
    #
    #     mask = labels == -1
    #     pcd_xyz, other_data = Filters.remove_points_by_mask(pcd_xyz, other_data, mask)
    #
    # pcd_xyz[:, 2] = v

    # 聚类去除离群点
    # if pcd_xyz.shape[0] >= 5:
    #     labels = Tools.dbscan(pcd_xyz, eps=epsilon, min_samples=min_samples)
    #     mask = labels == -1
    #     pcd_xyz, other_data = Filters.remove_points_by_mask(pcd_xyz, other_data, mask)

    left_point_id = other_data['pointinfo-id']  # 剩余点的id

    # 剩余点id的颜色为绿色，其他点为灰色
    colors = np.zeros((pcd_xyz_copy.shape[0], 3)) + [0.5, 0.5, 0.5]
    colors[left_point_id] = [0, 1, 0]
    other_data_copy['pointinfo-color'] = colors



    return pcd_xyz_copy, other_data_copy

if __name__ == '__main__':
    opt = Options().parse()
    # opt.window_height = 500
    # opt.window_width = 500

    opt.dataset = 'aeva'
    opt.scene_id = 5

    # opt.dataset = 'carla1'
    # opt.scene_id = 0

    visualizer = Visualizer(opt)
    scene = SceneLoader(opt)
    print(scene.frame_num)

    # pcd_xyz, other_data = scene.get_frame(7)
    # pcd_xyz, other_data = work(pcd_xyz, other_data)
    # visualizer.draw_points(pcd_xyz, other_data)

    visualizer.play_scene(scene, filter=work, delay_time=0, begin=0, end=-1)