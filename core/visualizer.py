# -*- coding: utf-8 -*-
import open3d as o3d
import numpy as np
import time
from tqdm import tqdm
from pynput import keyboard
import tkinter as tk
import threading
from easydict import EasyDict
from core import SceneLoader, Tools
from typing import Union

class Visualizer():
    def __init__(self, config: EasyDict):
        self.reset_config(config)
        self.pause = True
        self.frame_id = 0
        self.change_frame = False

    def __reset(self):
        self.pause = True
        self.frame_id = 0
        self.change_frame = False

    def __keybord_callback(self):
        # 检测键盘事件
        def on_press(key):
            # 空格键暂停
            if key == keyboard.Key.space:
                self.pause = not self.pause
                if self.pause:
                    print("已暂停,当前帧:", self.frame_id)
                else:
                    print("继续播放,起始帧:", self.frame_id)
            # 如果在暂停状态下按下←或↑键，后退一帧
            if (key == keyboard.Key.left or key == keyboard.Key.up) and self.pause:
                if self.frame_id == 0:
                    print("已经是第一帧了")
                    return
                self.frame_id -= 1
                self.change_frame = True
                print("当前帧:", self.frame_id)
            # 如果在暂停状态下按下→或↓键，前进一帧
            if (key == keyboard.Key.right or key == keyboard.Key.down) and self.pause:
                self.frame_id += 1
                self.change_frame = True
                print("当前帧:", self.frame_id)

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    def reset_config(self, config: EasyDict):
        '''
        重置可视化器配置
        :param config: EasyDict类型的配置
        :return:
        '''
        self.window_name = config.window_name
        self.window_height = config.window_height
        self.window_width = config.window_width
        self.window_left = config.window_left
        self.window_top = config.window_top
        self.background_color = config.background_color
        self.form1 = config.first_window.form
        self.point_size1 = config.first_window.point_size
        self.voxel_size1 = config.first_window.voxel_size
        self.octree_max_depth1 = config.first_window.octree_max_depth
        self.form2 = config.second_window.form
        self.point_size2 = config.second_window.point_size
        self.voxel_size2 = config.second_window.voxel_size
        self.octree_max_depth2 = config.second_window.octree_max_depth
        self.axis = config.axis
        self.init_camera_rpy = config.init_camera_rpy
        self.init_camera_T = config.init_camera_T
        self.camera_sync = config.camera_sync

    def draw_points(self, frame_data: Union[EasyDict, np.ndarray]):
        '''
        绘制点云
        :param frame_data: 点云数据，包含点云坐标和其他几何数据。或者直接传入点云坐标数组
        :return:
        '''
        if isinstance(frame_data, np.ndarray):
            # 如果传入的是点云坐标数组，则创建一个EasyDict对象
            frame_data = EasyDict({
                'pcd': {
                    'points': frame_data,
                    'colors': None,
                },
                'geometry': {
                    'arrows': [],
                    'spheres': [],
                    'boxes': [],
                },
                'pose': {
                    'R': None,
                    'T': None,
                }
            })


        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.window_name, height=self.window_height, width=self.window_width,
                            left=self.window_left, top=self.window_top)

        # 背景颜色
        vis.get_render_option().background_color = np.asarray(self.background_color)

        # 点云大小
        vis.get_render_option().point_size = self.point_size1

        # 点云
        pcd = o3d.geometry.PointCloud()
        if frame_data.pcd.points.shape[0] != 0:
            pcd.points = o3d.utility.Vector3dVector(frame_data.pcd.points)

            # 颜色
            if frame_data.pcd.colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(frame_data.pcd.colors)

            if self.form1 == "point":
                vis.add_geometry(pcd)
            elif self.form1 == "voxel":
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size1)
                vis.add_geometry(voxel_grid)
            elif self.form1 == "octree":
                octree = o3d.geometry.Octree(max_depth=self.octree_max_depth1)
                octree.convert_from_point_cloud(pcd, size_expand=0.01)
                vis.add_geometry(octree)
            else:
                assert False, "form参数错误, 只支持point, voxel, octree三种形式"

        # 三维框
        for box in frame_data.geometry.boxes:
            vis.add_geometry(box)

        # 箭头
        for arrow in frame_data.geometry.arrows:
            vis.add_geometry(arrow)

        # 球
        for sphere in frame_data.geometry.spheres:
            vis.add_geometry(sphere)

        # 坐标轴
        if self.axis != 0:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis, origin=[0, 0, 0])
            vis.add_geometry(ax)

        # text = o3d.t.geometry.TriangleMesh.create_text("Hello Open3D", depth=0.1).to_legacy()
        # text.paint_uniform_color((1, 0, 0))
        # vis.add_geometry(text)


        # 初始化视角
        vis.reset_view_point(True)
        # 修改相机初始位置
        if self.init_camera_rpy is not None and self.init_camera_T is not None:
            cam_params = o3d.camera.PinholeCameraParameters()

            R = Tools.Math.euler2mat(self.init_camera_rpy[0], self.init_camera_rpy[1], self.init_camera_rpy[2], degrees=True)
            T = np.array([self.init_camera_T[1], self.init_camera_T[0], self.init_camera_T[2]])

            Matrix = np.eye(4)
            Matrix[:3, :3] = R
            Matrix[:3, 3] = T
            cam_params.extrinsic = Matrix

            focal = 0.5

            cam_params.intrinsic.set_intrinsics(self.window_width,
                                                self.window_height,
                                                fx=self.window_width * focal,
                                                fy=self.window_width * focal,
                                                cx=self.window_width / 2,
                                                cy=self.window_height / 2)

            vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

        vis.run()
        vis.destroy_window()


    def draw_one_frame(self, scene: SceneLoader, frame_id: int, filter=None):
        '''
        绘制场景的某一帧
        :param scene: 场景加载器
        :param frame_id: 帧id
        :param filter: 过滤函数
        :return:
        '''
        frame_data = scene.get_frame(frame_id=frame_id, filter=filter)
        self.draw_points(frame_data)


    # def run_tkinter_controls(self, vis):
    #     def on_white_bg():
    #         vis.get_render_option().background_color = np.array([1, 1, 1])

    #     def on_black_bg():
    #         vis.get_render_option().background_color = np.array([0, 0, 0])

    #     root = tk.Tk()
    #     root.title("控制面板")
    #     tk.Button(root, text="白色背景", command=on_white_bg).pack()
    #     tk.Button(root, text="黑色背景", command=on_black_bg).pack()
    #     root.mainloop()


    def play_scene(self, scene: SceneLoader, frame_range=[0, -1], delay_time=0, filter=None):
        '''
        播放场景
        :param scene: 场景加载器
        :param frame_range: 帧范围, [begin, end], end为-1表示到最后一帧
        :param delay_time: 播放延迟时间
        :param filter: 过滤函数
        :return:
        '''
        self.__reset()
        begin = frame_range[0]
        end = frame_range[1] if frame_range[1] != -1 else scene.frame_num - 1

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.window_name, height=self.window_height, width=self.window_width,
                            left=self.window_left, top=self.window_top)

        # gui_thread = threading.Thread(target=self.run_tkinter_controls, args=(vis,))
        # gui_thread.start()

        # 背景颜色
        vis.get_render_option().background_color = np.asarray(self.background_color)

        # 点云大小
        vis.get_render_option().point_size = self.point_size1

        reset_view = False
        pcd = o3d.geometry.PointCloud()

        # 坐标轴
        if self.axis != 0:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis, origin=[0, 0, 0])

        self.__keybord_callback()

        self.frame_id = begin
        while self.frame_id <= end:
            print("frame_id:", self.frame_id)

            # 移除上一帧的几何对象
            vis.clear_geometries()

            # 坐标轴
            if self.axis != 0:
                vis.add_geometry(ax, reset_bounding_box=False)

            # 点云
            frame_data = scene.get_frame(frame_id=self.frame_id, filter=filter)
            if frame_data.pcd.points.shape[0] != 0:
                pcd.points = o3d.utility.Vector3dVector(frame_data.pcd.points)

                # 颜色
                if frame_data.pcd.colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(frame_data.pcd.colors)

                if self.form1 == "point":
                    vis.add_geometry(pcd, reset_bounding_box=False)
                elif self.form1 == "voxel":
                    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size1)
                    vis.add_geometry(voxel_grid, reset_bounding_box=False)
                elif self.form1 == "octree":
                    octree = o3d.geometry.Octree(max_depth=self.octree_max_depth1)
                    octree.convert_from_point_cloud(pcd, size_expand=0.01)
                    vis.add_geometry(octree, reset_bounding_box=False)
                else:
                    assert False, "form参数错误, 只支持point, voxel, octree三种形式"

            # 候选框
            for box in frame_data.geometry.boxes:
                vis.add_geometry(box, reset_bounding_box=False)

            # 箭头
            for arrow in frame_data.geometry.arrows:
                vis.add_geometry(arrow, reset_bounding_box=False)

            # 球
            for sphere in frame_data.geometry.spheres:
                vis.add_geometry(sphere, reset_bounding_box=False)

            if not reset_view:  # 初始化视角
                vis.reset_view_point(True)  # 重置视角

                # camera_pos = vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic
                # print("camera_pos:", camera_pos)
                # intrinsic = vis.get_view_control().convert_to_pinhole_camera_parameters().intrinsic.intrinsic_matrix
                # print("intrinsic:", intrinsic)

                # 修改相机初始位置
                if self.init_camera_rpy is not None and self.init_camera_T is not None:
                    cam_params = o3d.camera.PinholeCameraParameters()

                    R = Tools.Math.euler2mat(self.init_camera_rpy[0], self.init_camera_rpy[1], self.init_camera_rpy[2], degrees=True)
                    T = np.array([self.init_camera_T[1], self.init_camera_T[0], self.init_camera_T[2]])

                    Matrix = np.eye(4)
                    Matrix[:3, :3] = R
                    Matrix[:3, 3] = T
                    cam_params.extrinsic = Matrix

                    focal = 0.5  # 焦距
                    cam_params.intrinsic.set_intrinsics(self.window_width,
                                                        self.window_height,
                                                        fx=self.window_width * focal,
                                                        fy=self.window_width * focal,
                                                        cx=self.window_width / 2,
                                                        cy=self.window_height / 2)

                    vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

                reset_view = True

            # 更新渲染
            running = vis.poll_events()
            vis.update_renderer()
            if not running:
                break

            # 延时处理事件和渲染
            start_time = time.time()
            while time.time() - start_time < delay_time:
                running = vis.poll_events()
                vis.update_renderer()
                if not running:
                    break
            if not running:
                break

            # 暂停
            while self.pause:
                running = vis.poll_events()
                vis.update_renderer()
                if not running:
                    break
                if self.change_frame:
                    self.change_frame = False
                    break
            if not running:
                break

            if not self.pause:
                self.frame_id += 1

        vis.destroy_window()

    def draw_global_map(self, scene: SceneLoader, step=1):
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
            frame_data = scene.get_frame(frame_id=i)
            R, T = frame_data.pose.R, frame_data.pose.T
            if R is None or T is None:
                raise KeyError("请检查该数据集是否在load_frame中保存了位姿信息frame_data.pose.R和frame_data.pose.T")

            pcd_xyz_global = np.dot(frame_data.pcd.points, R.T) + T

            points_world.append(pcd_xyz_global)

        points_world = np.concatenate(points_world, axis=0)
        print("全局地图点数：", points_world.shape[0])

        self.draw_points(points_world)


    def compare_two_point_clouds(self, frame_data1: Union[EasyDict, np.ndarray], frame_data2: Union[EasyDict, np.ndarray]):
        '''
        双窗口比较两个点云
        :param frame_data1: 第一个点云数据，包含点云坐标和其他几何数据。或者直接传入点云坐标数组
        :param frame_data2: 第二个点云数据，包含点云坐标和其他几何数据。或者直接传入点云坐标数组
        :return:
        '''
        if isinstance(frame_data1, np.ndarray):
            frame_data1 = EasyDict({
                'pcd': {
                    'points': frame_data1,
                    'colors': None,
                },
                'geometry': {
                    'arrows': [],
                    'spheres': [],
                    'boxes': [],
                },
                'pose': {
                    'R': None,
                    'T': None,
                }
            })
        if isinstance(frame_data2, np.ndarray):
            frame_data2 = EasyDict({
                'pcd': {
                    'points': frame_data2,
                    'colors': None,
                },
                'geometry': {
                    'arrows': [],
                    'spheres': [],
                    'boxes': [],
                },
                'pose': {
                    'R': None,
                    'T': None,
                }
            })

        # 第一个窗口
        vis1 = o3d.visualization.Visualizer()
        vis1.create_window(window_name=self.window_name + '-1', height=self.window_height, width=self.window_width,
                            left=self.window_left, top=self.window_top)
        vis2 = o3d.visualization.Visualizer()
        vis2.create_window(window_name=self.window_name + '-2', height=self.window_height,
                            width=self.window_width,
                            left=self.window_left + self.window_width, top=self.window_top)

        # 背景颜色
        vis1.get_render_option().background_color = np.asarray(self.background_color)
        vis2.get_render_option().background_color = np.asarray(self.background_color)

        # 点云大小
        vis1.get_render_option().point_size = self.point_size1
        vis2.get_render_option().point_size = self.point_size2

        # 坐标轴
        if self.axis != 0:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis, origin=[0, 0, 0])  # 坐标轴
            vis1.add_geometry(ax)
            vis2.add_geometry(ax)

        # 点云
        pcd1 = o3d.geometry.PointCloud()
        if frame_data1.pcd.points.shape[0] != 0:
            pcd1.points = o3d.utility.Vector3dVector(frame_data1.pcd.points)
            # 颜色
            if frame_data1.pcd.colors is not None:
                pcd1.colors = o3d.utility.Vector3dVector(frame_data1.pcd.colors)

            if self.form1 == "point":
                vis1.add_geometry(pcd1)
            elif self.form1 == "voxel":
                voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd1, voxel_size=self.voxel_size1)
                vis1.add_geometry(voxel_grid1)
            elif self.form1 == "octree":
                octree1 = o3d.geometry.Octree(max_depth=self.octree_max_depth1)
                octree1.convert_from_point_cloud(pcd1, size_expand=0.01)
                vis1.add_geometry(octree1)
            else:
                assert False, "form1参数错误, 只支持point, voxel, octree三种形式"

        pcd2 = o3d.geometry.PointCloud()
        if frame_data2.pcd.points.shape[0] != 0:
            pcd2.points = o3d.utility.Vector3dVector(frame_data2.pcd.points)
            if frame_data2.pcd.colors is not None:
                pcd2.colors = o3d.utility.Vector3dVector(frame_data2.pcd.colors)

            if self.form2 == "point":
                vis2.add_geometry(pcd2)
            elif self.form2 == "voxel":
                voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd2, voxel_size=self.voxel_size2)
                vis2.add_geometry(voxel_grid2)
            elif self.form2 == "octree":
                octree2 = o3d.geometry.Octree(max_depth=self.octree_max_depth2)
                octree2.convert_from_point_cloud(pcd2, size_expand=0.01)
                vis2.add_geometry(octree2)
            else:
                assert False, "form2参数错误, 只支持point, voxel, octree三种形式"

        # 三维框
        for box in frame_data1.geometry.boxes:
            vis1.add_geometry(box)
        for box in frame_data2.geometry.boxes:
            vis2.add_geometry(box)

        # 箭头
        for arrow in frame_data1.geometry.arrows:
            vis1.add_geometry(arrow)
        for arrow in frame_data2.geometry.arrows:
            vis2.add_geometry(arrow)

        # 球
        for sphere in frame_data1.geometry.spheres:
            vis1.add_geometry(sphere)
        for sphere in frame_data2.geometry.spheres:
            vis2.add_geometry(sphere)

        # 初始化视角
        vis1.reset_view_point(True)
        vis2.reset_view_point(True)

        # 修改相机初始位置
        if self.init_camera_rpy is not None and self.init_camera_T is not None:
            cam_params = o3d.camera.PinholeCameraParameters()

            R = Tools.Math.euler2mat(self.init_camera_rpy[0], self.init_camera_rpy[1], self.init_camera_rpy[2], degrees=True)
            T = np.array([self.init_camera_T[1], self.init_camera_T[0], self.init_camera_T[2]])

            Matrix = np.eye(4)
            Matrix[:3, :3] = R
            Matrix[:3, 3] = T
            cam_params.extrinsic = Matrix

            focal = 0.5

            cam_params.intrinsic.set_intrinsics(self.window_width,
                                                self.window_height,
                                                fx=self.window_width * focal,
                                                fy=self.window_width * focal,
                                                cx=self.window_width / 2,
                                                cy=self.window_height / 2)

            vis1.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)


        running1 = True
        running2 = True

        while running1 and running2:
            if self.camera_sync:
                cam1 = vis1.get_view_control().convert_to_pinhole_camera_parameters()
                vis2.get_view_control().convert_from_pinhole_camera_parameters(cam1)
            running1 = vis1.poll_events()
            if self.camera_sync:
                cam2 = vis2.get_view_control().convert_to_pinhole_camera_parameters()
                vis1.get_view_control().convert_from_pinhole_camera_parameters(cam2)
            running2 = vis2.poll_events()

            vis1.update_renderer()
            vis2.update_renderer()

        vis1.destroy_window()
        vis2.destroy_window()


    def compare_one_frame(self, scene: SceneLoader, frame_id: int, filter1=None, filter2=None):
        '''
        比较两个过滤器的结果
        :param scene: 场景加载器
        :param frame_id: 帧id
        :param filter1: 过滤函数1
        :param filter2: 过滤函数2
        :return:
        '''

        frame_data1 = scene.get_frame(frame_id=frame_id, filter=filter1)
        frame_data2 = scene.get_frame(frame_id=frame_id, filter=filter2)
        self.compare_two_point_clouds(frame_data1, frame_data2)


    def compare_scene(self, scene: SceneLoader, filter1=None, filter2=None, frame_range=[0, -1], delay_time=0):
        '''
        播放场景并比较两个过滤函数的结果
        :param scene: 场景加载器
        :param filter1: 过滤函数1
        :param filter2: 过滤函数2
        :param frame_range: 帧范围, [begin, end], end为-1表示到最后一帧
        :param delay_time: 播放延迟时间
        :return:
        '''
        self.__reset()
        begin = frame_range[0]
        end = frame_range[1] if frame_range[1] != -1 else scene.frame_num - 1

        reset_view = False

        vis1 = o3d.visualization.Visualizer()
        vis2 = o3d.visualization.Visualizer()
        vis1.create_window(window_name=self.window_name + '-1', height=self.window_height, width=self.window_width,
                            left=self.window_left, top=self.window_top)
        vis2.create_window(window_name=self.window_name + '-2', height=self.window_height, width=self.window_width,
                            left=self.window_left + self.window_width, top=self.window_top)

        # 背景颜色
        vis1.get_render_option().background_color = np.asarray(self.background_color)
        vis2.get_render_option().background_color = np.asarray(self.background_color)

        # 点云大小
        vis1.get_render_option().point_size = self.point_size1
        vis2.get_render_option().point_size = self.point_size2

        # 点云
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()

        # 坐标轴
        if self.axis != 0:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis, origin=[0, 0, 0])

        self.__keybord_callback()

        self.frame_id = begin
        while self.frame_id <= end:
            print("frame_id:", self.frame_id)

            # # 移除上一帧的几何对象
            vis1.clear_geometries()
            vis2.clear_geometries()

            # 坐标轴
            if self.axis != 0:
                vis1.add_geometry(ax, reset_bounding_box=False)
                vis2.add_geometry(ax, reset_bounding_box=False)

            # 点云
            frame_data1 = scene.get_frame(frame_id=self.frame_id, filter=filter1)
            if frame_data1.pcd.points.shape[0] != 0:
                pcd1.points = o3d.utility.Vector3dVector(frame_data1.pcd.points)

                # 颜色
                if frame_data1.pcd.colors is not None:
                    pcd1.colors = o3d.utility.Vector3dVector(frame_data1.pcd.colors)

                if self.form1 == "point":
                    vis1.add_geometry(pcd1, reset_bounding_box=False)
                elif self.form1 == "voxel":
                    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd1, voxel_size=self.voxel_size1)
                    vis1.add_geometry(voxel_grid1, reset_bounding_box=False)
                elif self.form1 == "octree":
                    octree1 = o3d.geometry.Octree(max_depth=self.octree_max_depth1)
                    octree1.convert_from_point_cloud(pcd1, size_expand=0.01)
                    vis1.add_geometry(octree1, reset_bounding_box=False)
                else:
                    assert False, "form1参数错误, 只支持point, voxel, octree三种形式"


            frame_data2 = scene.get_frame(frame_id=self.frame_id, filter=filter2)
            if frame_data2.pcd.points.shape[0] != 0:
                pcd2.points = o3d.utility.Vector3dVector(frame_data2.pcd.points)
                if frame_data2.pcd.colors is not None:
                    pcd2.colors = o3d.utility.Vector3dVector(frame_data2.pcd.colors)

                if self.form2 == "point":
                    vis2.add_geometry(pcd2, reset_bounding_box=False)
                elif self.form2 == "voxel":
                    voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd2, voxel_size=self.voxel_size2)
                    vis2.add_geometry(voxel_grid2, reset_bounding_box=False)
                elif self.form2 == "octree":
                    octree2 = o3d.geometry.Octree(max_depth=self.octree_max_depth2)
                    octree2.convert_from_point_cloud(pcd2, size_expand=0.01)
                    vis2.add_geometry(octree2, reset_bounding_box=False)
                else:
                    assert False, "form2参数错误, 只支持point, voxel, octree三种形式"

            # 候选框
            for box in frame_data1.geometry.boxes:
                vis1.add_geometry(box, reset_bounding_box=False)
            for box in frame_data2.geometry.boxes:
                vis2.add_geometry(box, reset_bounding_box=False)

            # 箭头
            for arrow in frame_data1.geometry.arrows:
                vis1.add_geometry(arrow, reset_bounding_box=False)
            for arrow in frame_data2.geometry.arrows:
                vis2.add_geometry(arrow, reset_bounding_box=False)

            # 球
            for sphere in frame_data1.geometry.spheres:
                vis1.add_geometry(sphere, reset_bounding_box=False)
            for sphere in frame_data2.geometry.spheres:
                vis2.add_geometry(sphere, reset_bounding_box=False)

            if not reset_view:  # 初始化视角
                vis1.reset_view_point(True)  # 重置视角
                vis2.reset_view_point(True)

                # 修改相机初始位置
                if self.init_camera_rpy is not None and self.init_camera_T is not None:
                    cam_params = o3d.camera.PinholeCameraParameters()

                    R = Tools.Math.euler2mat(self.init_camera_rpy[0], self.init_camera_rpy[1], self.init_camera_rpy[2], degrees=True)
                    T = np.array([self.init_camera_T[1], self.init_camera_T[0], self.init_camera_T[2]])

                    Matrix = np.eye(4)
                    Matrix[:3, :3] = R
                    Matrix[:3, 3] = T
                    cam_params.extrinsic = Matrix

                    focal = 0.5  # 焦距
                    cam_params.intrinsic.set_intrinsics(self.window_width,
                                                        self.window_height,
                                                        fx=self.window_width * focal,
                                                        fy=self.window_width * focal,
                                                        cx=self.window_width / 2,
                                                        cy=self.window_height / 2)

                    vis1.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

                reset_view = True

            if self.camera_sync:
                cam1 = vis1.get_view_control().convert_to_pinhole_camera_parameters()
                vis2.get_view_control().convert_from_pinhole_camera_parameters(cam1)
            running1 = vis1.poll_events()
            if self.camera_sync:
                cam2 = vis2.get_view_control().convert_to_pinhole_camera_parameters()
                vis1.get_view_control().convert_from_pinhole_camera_parameters(cam2)
            running2 = vis2.poll_events()
            vis1.update_renderer()
            vis2.update_renderer()
            if not running1 or not running2:
                break

            # 延时处理事件和渲染
            start_time = time.time()
            while time.time() - start_time < delay_time:
                if self.camera_sync:
                    cam1 = vis1.get_view_control().convert_to_pinhole_camera_parameters()
                    vis2.get_view_control().convert_from_pinhole_camera_parameters(cam1)
                running1 = vis1.poll_events()
                if self.camera_sync:
                    cam2 = vis2.get_view_control().convert_to_pinhole_camera_parameters()
                    vis1.get_view_control().convert_from_pinhole_camera_parameters(cam2)
                running2 = vis2.poll_events()
                vis1.update_renderer()
                vis2.update_renderer()
                if not running1 or not running2:
                    break
            if not running1 or not running2:
                break

            # 处理键盘事件
            while self.pause:
                if self.camera_sync:
                    cam1 = vis1.get_view_control().convert_to_pinhole_camera_parameters()
                    vis2.get_view_control().convert_from_pinhole_camera_parameters(cam1)
                running1 = vis1.poll_events()
                if self.camera_sync:
                    cam2 = vis2.get_view_control().convert_to_pinhole_camera_parameters()
                    vis1.get_view_control().convert_from_pinhole_camera_parameters(cam2)
                running2 = vis2.poll_events()
                vis1.update_renderer()
                vis2.update_renderer()
                if not running1 or not running2:
                    break
                if self.change_frame:
                    self.change_frame = False
                    break
            if not running1 or not running2:
                break

            if not self.pause:
                self.frame_id += 1

        vis1.destroy_window()
        vis2.destroy_window()


    def headless(self, scene: SceneLoader, frame_range=[0, -1], filter=None):
            '''
            无头模式播放场景
            :param scene: 场景加载器
            :param frame_range: 帧范围, [begin, end], end为-1表示到最后一帧
            :param filter: 过滤函数
            :return:
            '''
            self.__reset()
            begin = frame_range[0]
            end = frame_range[1] if frame_range[1] != -1 else scene.frame_num - 1

            self.frame_id = begin
            while self.frame_id <= end:
                print("frame_id:", self.frame_id)
                frame_data = scene.get_frame(frame_id=self.frame_id, filter=filter)
                self.frame_id += 1