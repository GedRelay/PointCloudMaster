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
from utils.tools import Tools
from tqdm import tqdm


class Visualizer():
    def __init__(self, opt):
        self.opt = opt

    def draw_points(self, points, other_data=None, axis=5, init_camera_rpy=None,
                   init_camera_T=None):
        '''
        绘制点云
        :param points: 点云 N*3
        :param other_data: 其他数据, 可能包含颜色color, 候选框bbox, 箭头arrows等
        :param axis: 坐标轴大小, None表示不绘制坐标轴
        :param init_camera_rpy: 相机初始姿态 [roll, pitch, yaw]
        :param init_camera_T: 相机初始位置 [x, y, z]
        :return:
        '''

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.opt.window_name, height=self.opt.window_height, width=self.opt.window_width,
                            left=self.opt.window_left, top=self.opt.window_top)

        # 背景颜色
        vis.get_render_option().background_color = np.asarray(self.opt.background_color)

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

        # 初始化视角
        vis.reset_view_point(True)
        # 修改相机初始位置
        if init_camera_rpy is not None and init_camera_T is not None:
            cam_params = o3d.camera.PinholeCameraParameters()

            R = Tools.euler2mat(init_camera_rpy[0], init_camera_rpy[1], init_camera_rpy[2], degrees=True)
            T = np.array([init_camera_T[1], init_camera_T[0], init_camera_T[2]])

            Matrix = np.eye(4)
            Matrix[:3, :3] = R
            Matrix[:3, 3] = T
            cam_params.extrinsic = Matrix

            focal = 0.5

            cam_params.intrinsic.set_intrinsics(self.opt.window_width,
                                                self.opt.window_height,
                                                fx=self.opt.window_width * focal,
                                                fy=self.opt.window_width * focal,
                                                cx=self.opt.window_width / 2,
                                                cy=self.opt.window_height / 2)

            vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

        vis.run()
        vis.destroy_window()


    def draw_one_frame(self, scene, frame_id, axis=5, filter=None, init_camera_rpy=None, init_camera_T=None):
        '''
        绘制场景的某一帧
        :param scene: 场景加载器
        :param frame_id: 帧id
        :param axis: 坐标轴大小, None表示不绘制坐标轴
        :param filter: 过滤函数
        :param init_camera_rpy: 相机初始姿态 [roll, pitch, yaw]
        :param init_camera_T: 相机初始位置 [x, y, z]
        :return:
        '''

        pcd_xyz, other_data = scene.get_frame(frame_id=frame_id, filter=filter)

        self.draw_points(pcd_xyz, other_data, axis=axis, init_camera_rpy=init_camera_rpy, init_camera_T=init_camera_T)



    def play_scene(self, scene, begin=0, end=-1, delay_time=0.1, axis=5, filter=None, init_camera_rpy=None,
                   init_camera_T=None):
        '''
        播放场景
        :param scene: 场景加载器
        :param begin: 开始帧
        :param end: 结束帧, -1表示最后一帧
        :param delay_time: 延迟时间
        :param axis: 坐标轴大小, None表示不绘制坐标轴
        :param filter: 过滤函数
        :param init_camera_rpy: 相机初始姿态 [roll, pitch, yaw]
        :param init_camera_T: 相机初始位置 [x, y, z]
        :return:
        '''

        if end == -1:
            end = scene.frame_num

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.opt.window_name, height=self.opt.window_height, width=self.opt.window_width,
                            left=self.opt.window_left, top=self.opt.window_top)

        # 背景颜色
        vis.get_render_option().background_color = np.asarray(self.opt.background_color)

        reset_view = False

        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        # 坐标轴
        if axis is not None:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis, origin=[0, 0, 0])
            vis.add_geometry(ax)

        # 追踪已添加的几何对象
        geometries = []

        running = True

        for i in range(begin, end):
            # print("frame_id:", i)
            # print("frame_name:", scene.dataset_loader.filenames[i])

            # 移除上一帧的几何对象
            for geometry in geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)
            geometries.clear()

            pcd_xyz, other_data = scene.get_frame(frame_id=i, filter=filter)
            pcd.points = o3d.utility.Vector3dVector(pcd_xyz)

            # 颜色
            if 'pointinfo-color' in other_data.keys():
                pcd.colors = o3d.utility.Vector3dVector(other_data['pointinfo-color'])

            # 候选框
            if 'geometry-bboxes' in other_data.keys():
                for bbox in other_data['geometry-bboxes']:
                    vis.add_geometry(bbox, reset_bounding_box=False)
                    geometries.append(bbox)

            # 箭头
            if 'geometry-arrows' in other_data.keys():
                for arrow in other_data['geometry-arrows']:
                    vis.add_geometry(arrow, reset_bounding_box=False)
                    geometries.append(arrow)

            # 球
            if 'geometry-spheres' in other_data.keys():
                for sphere in other_data['geometry-spheres']:
                    vis.add_geometry(sphere, reset_bounding_box=False)
                    geometries.append(sphere)

            vis.update_geometry(pcd)

            running = vis.poll_events()
            if not running:
                break
            vis.update_renderer()

            if not reset_view:  # 初始化视角
                vis.reset_view_point(True)  # 重置视角

                # camera_pos = vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic
                # print("camera_pos:", camera_pos)
                # intrinsic = vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic.intrinsic_matrix
                # print("intrinsic:", intrinsic)

                # 修改相机初始位置
                if init_camera_rpy is not None and init_camera_T is not None:
                    cam_params = o3d.camera.PinholeCameraParameters()

                    R = Tools.euler2mat(init_camera_rpy[0], init_camera_rpy[1], init_camera_rpy[2], degrees=True)
                    T = np.array([init_camera_T[1], init_camera_T[0], init_camera_T[2]])

                    Matrix = np.eye(4)
                    Matrix[:3, :3] = R
                    Matrix[:3, 3] = T
                    cam_params.extrinsic = Matrix

                    focal = 0.5  # 焦距
                    cam_params.intrinsic.set_intrinsics(self.opt.window_width,
                                                        self.opt.window_height,
                                                        fx=self.opt.window_width * focal,
                                                        fy=self.opt.window_width * focal,
                                                        cx=self.opt.window_width / 2,
                                                        cy=self.opt.window_height / 2)

                    vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

                reset_view = True

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


    def compare_two_point_clouds(self, pcd_xyz1, pcd_xyz2, other_data1=None, other_data2=None, axis=5,
                                 init_camera_rpy=None, init_camera_T=None):
        '''
        比较两个点云, 同步视角显示
        :param pcd_xyz1: 点云1
        :param pcd_xyz2: 点云2
        :param other_data1: 其他数据1, 可能包含颜色color, 候选框bbox, 箭头arrows等
        :param other_data2: 其他数据2, 可能包含颜色color, 候选框bbox, 箭头arrows等
        :param axis: 坐标轴大小, None表示不绘制坐标轴
        :param init_camera_rpy: 相机初始姿态 [roll, pitch, yaw]
        :param init_camera_T: 相机初始位置 [x, y, z]
        :return:
        '''

        # 第一个窗口
        vis1 = o3d.visualization.Visualizer()
        vis1.create_window(window_name=self.opt.window_name + '-1', height=self.opt.window_height, width=self.opt.window_width,
                            left=self.opt.window_left, top=self.opt.window_top)

        # 背景颜色
        vis1.get_render_option().background_color = np.asarray(self.opt.background_color)

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcd_xyz1)

        # 颜色
        if other_data1 is not None and 'pointinfo-color' in other_data1.keys():
            pcd1.colors = o3d.utility.Vector3dVector(other_data1['pointinfo-color'])

        # 候选框
        if other_data1 is not None and 'geometry-bboxes' in other_data1.keys():
            for bbox in other_data1['geometry-bboxes']:
                vis1.add_geometry(bbox)

        # 箭头
        if other_data1 is not None and 'geometry-arrows' in other_data1.keys():
            for arrow in other_data1['geometry-arrows']:
                vis1.add_geometry(arrow)

        # 球
        if other_data1 is not None and 'geometry-spheres' in other_data1.keys():
            for sphere in other_data1['geometry-spheres']:
                vis1.add_geometry(sphere)

        # 坐标轴
        if axis is not None:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis, origin=[0, 0, 0])  # 坐标轴
            vis1.add_geometry(ax)

        # text = o3d.t.geometry.TriangleMesh.create_text("Hello Open3D", depth=0.1).to_legacy()
        # text.paint_uniform_color((1, 0, 0))
        # vis.add_geometry(text)

        vis1.add_geometry(pcd1)

        # 初始化视角
        vis1.reset_view_point(True)
        # 修改相机初始位置
        if init_camera_rpy is not None and init_camera_T is not None:
            cam_params = o3d.camera.PinholeCameraParameters()

            R = Tools.euler2mat(init_camera_rpy[0], init_camera_rpy[1], init_camera_rpy[2], degrees=True)
            T = np.array([init_camera_T[1], init_camera_T[0], init_camera_T[2]])

            Matrix = np.eye(4)
            Matrix[:3, :3] = R
            Matrix[:3, 3] = T
            cam_params.extrinsic = Matrix

            focal = 0.5

            cam_params.intrinsic.set_intrinsics(self.opt.window_width,
                                                self.opt.window_height,
                                                fx=self.opt.window_width * focal,
                                                fy=self.opt.window_width * focal,
                                                cx=self.opt.window_width / 2,
                                                cy=self.opt.window_height / 2)

            vis1.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

        # 第二个窗口
        vis2 = o3d.visualization.Visualizer()
        vis2.create_window(window_name=self.opt.window_name + '-2', height=self.opt.window_height, width=self.opt.window_width,
                            left=self.opt.window_left + self.opt.window_width, top=self.opt.window_top)

        # 背景颜色
        vis2.get_render_option().background_color = np.asarray(self.opt.background_color)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pcd_xyz2)

        # 颜色
        if other_data2 is not None and 'pointinfo-color' in other_data2.keys():
            pcd2.colors = o3d.utility.Vector3dVector(other_data2['pointinfo-color'])

        # 候选框
        if other_data2 is not None and 'geometry-bboxes' in other_data2.keys():
            for bbox in other_data2['geometry-bboxes']:
                vis2.add_geometry(bbox)

        # 箭头
        if other_data2 is not None and 'geometry-arrows' in other_data2.keys():
            for arrow in other_data2['geometry-arrows']:
                vis2.add_geometry(arrow)

        # 球
        if other_data2 is not None and 'geometry-spheres' in other_data2.keys():
            for sphere in other_data2['geometry-spheres']:
                vis2.add_geometry(sphere)

        # 坐标轴
        if axis is not None:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis, origin=[0, 0, 0])
            vis2.add_geometry(ax)

        vis2.add_geometry(pcd2)

        # 初始化视角
        vis2.reset_view_point(True)

        running1 = True
        running2 = True

        while running1 and running2:
            # 更新渲染, 保持两个窗口视角一致
            cam1 = vis1.get_view_control().convert_to_pinhole_camera_parameters()
            vis2.get_view_control().convert_from_pinhole_camera_parameters(cam1)
            running1 = vis1.poll_events()

            cam2 = vis2.get_view_control().convert_to_pinhole_camera_parameters()
            vis1.get_view_control().convert_from_pinhole_camera_parameters(cam2)
            running2 = vis2.poll_events()

            vis1.update_renderer()
            vis2.update_renderer()

        vis1.destroy_window()
        vis2.destroy_window()


    def compare_one_frame(self, scene, frame_id, axis=5, filter1=None, filter2=None,
                            init_camera_rpy=None, init_camera_T=None):
        '''
        比较两个过滤器的结果，同步视角显示
        :param scene: 场景加载器
        :param frame_id: 帧id
        :param axis: 坐标轴大小, None表示不绘制坐标轴
        :param filter1: 过滤函数1
        :param filter2: 过滤函数2
        :param init_camera_rpy: 相机初始姿态 [roll, pitch, yaw]
        :param init_camera_T: 相机初始位置 [x, y, z]
        :return:
        '''

        pcd_xyz1, other_data1 = scene.get_frame(frame_id=frame_id, filter=filter1)
        pcd_xyz2, other_data2 = scene.get_frame(frame_id=frame_id, filter=filter2)

        self.compare_two_point_clouds(pcd_xyz1, pcd_xyz2, other_data1, other_data2, axis=axis,
                                        init_camera_rpy=init_camera_rpy, init_camera_T=init_camera_T)


    def compare_scene(self, scene, filter1=None, filter2=None, delay_time=0.1, begin=0, end=-1, axis=5,
                        init_camera_rpy=None, init_camera_T=None):
        '''
        播放场景并比较两个过滤函数的结果, 同步视角显示
        :param scene: 场景加载器
        :param filter1: 过滤函数1
        :param filter2: 过滤函数2
        :param delay_time: 延迟时间
        :param begin: 开始帧, 从0开始
        :param end: 结束帧, -1表示最后一帧
        :param axis: 坐标轴大小, None表示不绘制坐标轴
        :param init_camera_rpy: 相机初始姿态 [roll, pitch, yaw]
        :param init_camera_T: 相机初始位置 [x, y, z]
        :return:
        '''
        if end == -1:
            end = scene.frame_num

        reset_view = False

        # 第一个窗口
        vis1 = o3d.visualization.Visualizer()
        vis2 = o3d.visualization.Visualizer()
        vis1.create_window(window_name=self.opt.window_name + '-1', height=self.opt.window_height, width=self.opt.window_width,
                            left=self.opt.window_left, top=self.opt.window_top)
        vis2.create_window(window_name=self.opt.window_name + '-2', height=self.opt.window_height, width=self.opt.window_width,
                            left=self.opt.window_left + self.opt.window_width, top=self.opt.window_top)

        # 背景颜色
        vis1.get_render_option().background_color = np.asarray(self.opt.background_color)
        vis2.get_render_option().background_color = np.asarray(self.opt.background_color)

        # 点云
        pcd1 = o3d.geometry.PointCloud()
        vis1.add_geometry(pcd1)
        pcd2 = o3d.geometry.PointCloud()
        vis2.add_geometry(pcd2)

        # 坐标轴
        if axis is not None:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis, origin=[0, 0, 0])
            vis1.add_geometry(ax)
            vis2.add_geometry(ax)

        # 追踪已添加的几何对象
        geometries1 = []
        geometries2 = []

        for i in range(begin, end):
            # print("frame_id:", i)
            # print("frame_name:", scene.dataset_loader.filenames[i])

            # 移除上一帧的几何对象
            for geometry in geometries1:
                vis1.remove_geometry(geometry, reset_bounding_box=False)
            geometries1.clear()
            for geometry in geometries2:
                vis2.remove_geometry(geometry, reset_bounding_box=False)
            geometries2.clear()

            pcd_xyz1, other_data1 = scene.get_frame(frame_id=i, filter=filter1)
            pcd1.points = o3d.utility.Vector3dVector(pcd_xyz1)
            pcd_xyz2, other_data2 = scene.get_frame(frame_id=i, filter=filter2)
            pcd2.points = o3d.utility.Vector3dVector(pcd_xyz2)

            # 颜色
            if 'pointinfo-color' in other_data1.keys():
                pcd1.colors = o3d.utility.Vector3dVector(other_data1['pointinfo-color'])
            if 'pointinfo-color' in other_data2.keys():
                pcd2.colors = o3d.utility.Vector3dVector(other_data2['pointinfo-color'])

            # 候选框
            if 'geometry-bboxes' in other_data1.keys():
                for bbox in other_data1['geometry-bboxes']:
                    vis1.add_geometry(bbox, reset_bounding_box=False)
                    geometries1.append(bbox)
            if 'geometry-bboxes' in other_data2.keys():
                for bbox in other_data2['geometry-bboxes']:
                    vis2.add_geometry(bbox, reset_bounding_box=False)
                    geometries2.append(bbox)

            # 箭头
            if 'geometry-arrows' in other_data1.keys():
                for arrow in other_data1['geometry-arrows']:
                    vis1.add_geometry(arrow, reset_bounding_box=False)
                    geometries1.append(arrow)
            if 'geometry-arrows' in other_data2.keys():
                for arrow in other_data2['geometry-arrows']:
                    vis2.add_geometry(arrow, reset_bounding_box=False)
                    geometries2.append(arrow)

            # 球
            if 'geometry-spheres' in other_data1.keys():
                for sphere in other_data1['geometry-spheres']:
                    vis1.add_geometry(sphere, reset_bounding_box=False)
                    geometries1.append(sphere)
            if 'geometry-spheres' in other_data2.keys():
                for sphere in other_data2['geometry-spheres']:
                    vis2.add_geometry(sphere, reset_bounding_box=False)
                    geometries2.append(sphere)

            vis1.update_geometry(pcd1)
            vis2.update_geometry(pcd2)

            if not reset_view:  # 初始化视角
                vis1.reset_view_point(True)  # 重置视角
                vis2.reset_view_point(True)

                # 修改相机初始位置
                if init_camera_rpy is not None and init_camera_T is not None:
                    cam_params = o3d.camera.PinholeCameraParameters()

                    R = Tools.euler2mat(init_camera_rpy[0], init_camera_rpy[1], init_camera_rpy[2], degrees=True)
                    T = np.array([init_camera_T[1], init_camera_T[0], init_camera_T[2]])

                    Matrix = np.eye(4)
                    Matrix[:3, :3] = R
                    Matrix[:3, 3] = T
                    cam_params.extrinsic = Matrix

                    focal = 0.5  # 焦距
                    cam_params.intrinsic.set_intrinsics(self.opt.window_width,
                                                        self.opt.window_height,
                                                        fx=self.opt.window_width * focal,
                                                        fy=self.opt.window_width * focal,
                                                        cx=self.opt.window_width / 2,
                                                        cy=self.opt.window_height / 2)

                    vis1.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

                reset_view = True

            cam1 = vis1.get_view_control().convert_to_pinhole_camera_parameters()
            vis2.get_view_control().convert_from_pinhole_camera_parameters(cam1)
            running1 = vis1.poll_events()

            cam2 = vis2.get_view_control().convert_to_pinhole_camera_parameters()
            vis1.get_view_control().convert_from_pinhole_camera_parameters(cam2)
            running2 = vis2.poll_events()

            if not running1 or not running2:
                break

            vis1.update_renderer()
            vis2.update_renderer()

            # 延时处理事件和渲染
            start_time = time.time()
            while time.time() - start_time < delay_time:
                cam1 = vis1.get_view_control().convert_to_pinhole_camera_parameters()
                vis2.get_view_control().convert_from_pinhole_camera_parameters(cam1)
                running1 = vis1.poll_events()
                cam2 = vis2.get_view_control().convert_to_pinhole_camera_parameters()
                vis1.get_view_control().convert_from_pinhole_camera_parameters(cam2)
                running2 = vis2.poll_events()

                if not running1 or not running2:
                    break

                vis1.update_renderer()
                vis2.update_renderer()

            if not running1 or not running2:
                break

        vis1.destroy_window()
        vis2.destroy_window()

