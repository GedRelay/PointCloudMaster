# -*- coding: utf-8 -*-
"""
@Time        :  2024/6/4 18:17
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  visualizer 可视化工具
"""
import open3d as o3d
import numpy as np
import time
from tqdm import tqdm

class Visualizer():
    def __init__(self, opt):
        self.opt = opt

    def draw_points(self, points, other_data=None, axis=5):
        '''
        绘制点云
        :param points: 点云 N*3
        :param other_data: 其他数据, 可能包含颜色color, 候选框bbox, 箭头arrows等
        :param axis: 坐标轴大小
        :return:
        '''

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.opt.window_name, height=self.opt.window_height, width=self.opt.window_width)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 颜色
        if other_data is not None and 'pointinfo-color' in other_data.keys():
            pcd.colors = o3d.utility.Vector3dVector(other_data['pointinfo-color'])

        # 候选框
        if other_data is not None and 'geometry-bboxes' in other_data.keys():
            for bbox in other_data['geometry-bboxes']:
                vis.add_geometry(bbox)

        # 箭头
        if other_data is not None and 'geometry-arrows' in other_data.keys():
            for arrow in other_data['geometry-arrows']:
                vis.add_geometry(arrow)

        # 球
        if other_data is not None and 'geometry-spheres' in other_data.keys():
            for sphere in other_data['geometry-spheres']:
                vis.add_geometry(sphere)

        # 坐标轴
        if axis is not None:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis, origin=[0, 0, 0])  # 坐标轴
            vis.add_geometry(ax)

        # text = o3d.t.geometry.TriangleMesh.create_text("Hello Open3D", depth=0.1).to_legacy()
        # text.paint_uniform_color((1, 0, 0))
        # vis.add_geometry(text)

        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()


    def play_scene(self, scene, begin=0, end=-1, delay_time=0.1, axis=5, filter=None):
        '''
        播放场景
        :param scene: 场景加载器
        :param begin: 开始帧
        :param end: 结束帧, -1表示最后一帧
        :param delay_time: 延迟时间
        :param axis: 坐标轴大小
        :param filter: 过滤函数
        :return:
        '''

        if end == -1:
            end = scene.frame_num

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.opt.window_name, height=self.opt.window_height, width=self.opt.window_width)
        to_reset = True

        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        # 坐标轴
        if axis is not None:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis, origin=[0, 0, 0])
            vis.add_geometry(ax)

        # 追踪已添加的几何对象
        geometries = []

        for i in range(begin, end):
            # print("frame_id:", i)
            # print("frame_name:", scene.dataset_loader.filenames[i])

            # 移除上一帧的几何对象
            for geometry in geometries:
                vis.remove_geometry(geometry)
            geometries.clear()

            pcd_xyz, other_data = scene.get_frame(frame_id=i, filter=filter)
            pcd.points = o3d.utility.Vector3dVector(pcd_xyz)

            # 颜色
            if 'pointinfo-color' in other_data.keys():
                pcd.colors = o3d.utility.Vector3dVector(other_data['pointinfo-color'])

            # 候选框
            if 'geometry-bboxes' in other_data.keys():
                for bbox in other_data['geometry-bboxes']:
                    vis.add_geometry(bbox)
                    geometries.append(bbox)

            # 箭头
            if 'geometry-arrows' in other_data.keys():
                for arrow in other_data['geometry-arrows']:
                    vis.add_geometry(arrow)
                    geometries.append(arrow)

            # 球
            if 'geometry-spheres' in other_data.keys():
                for sphere in other_data['geometry-spheres']:
                    vis.add_geometry(sphere)
                    geometries.append(sphere)

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            if to_reset:
                vis.reset_view_point(True)
                to_reset = False

            # 延时处理事件和渲染
            start_time = time.time()
            while time.time() - start_time < delay_time:
                vis.poll_events()
                vis.update_renderer()

        vis.destroy_window()

    def draw_global_map(self, scene, step=1, axis=5):
        '''
        绘制全局地图
        :param scene: 场景加载器
        :param step: 采样步长，每隔step帧采样一次
        :return:
        '''
        points_world = []

        bar = tqdm(range(0, scene.frame_num, step))
        for i in bar:
            bar.set_description("加载中")
            pcd_xyz, _ = scene.get_frame(frame_id=i)
            R, T = scene.get_pose(frame_id=i)

            pcd_xyz_global = np.dot(pcd_xyz, R.T) + T

            points_world.append(pcd_xyz_global)

        points_world = np.concatenate(points_world, axis=0)
        print("全局地图点数：", points_world.shape[0])

        self.draw_points(points_world, axis=axis)