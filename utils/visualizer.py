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
from pynput import keyboard


class Visualizer():
    def __init__(self, opt=None, window_name="Visualizer", window_height=540, window_width=960,
                 window_left=50, window_top=50, background_color=[1, 1, 1]):
        # 注意如果opt不为None时后面的参数都不可用
        self.window_name = window_name
        self.window_height = window_height
        self.window_width = window_width
        self.window_left = window_left
        self.window_top = window_top
        self.background_color = background_color
        self.__load_settings_by_opt(opt)

        self.pause = True
        self.frame_id = 0
        self.change_frame = False

    def __load_settings_by_opt(self, opt):
        if opt is not None:
            self.window_name = opt.window_name
            self.window_height = opt.window_height
            self.window_width = opt.window_width
            self.window_left = opt.window_left
            self.window_top = opt.window_top
            self.background_color = opt.background_color

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


    def draw_points(self, points, other_data=None, form="point", point_size=2.0, voxel_size=0.5, octree_max_depth=8,
                    axis=5, init_camera_rpy=None, init_camera_T=None):
        '''
        绘制点云
        :param points: 点云 N*3
        :param other_data: 其他数据, 可能包含颜色color, 候选框bbox, 箭头arrows等
        :param form: 绘制形式, point表示点云, voxel表示体素化, octree表示八叉树, empty表示不绘制
        :param point_size: 点云大小，当form为point时有效
        :param voxel_size: 体素大小，当form为voxel时有效
        :param octree_max_depth: 八叉树最大深度，当form为octree时有效
        :param axis: 坐标轴大小, None表示不绘制坐标轴
        :param init_camera_rpy: 相机初始姿态 [roll, pitch, yaw]
        :param init_camera_T: 相机初始位置 [x, y, z]
        :return:
        '''

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.window_name, height=self.window_height, width=self.window_width,
                            left=self.window_left, top=self.window_top)

        # 背景颜色
        vis.get_render_option().background_color = np.asarray(self.background_color)

        # 点云大小
        vis.get_render_option().point_size = point_size

        # 点云
        pcd = o3d.geometry.PointCloud()
        if points.shape[0] != 0:
            pcd.points = o3d.utility.Vector3dVector(points)

            # 颜色
            if other_data is not None and 'pointinfo-color' in other_data.keys():
                pcd.colors = o3d.utility.Vector3dVector(other_data['pointinfo-color'])

            if form == "point":
                vis.add_geometry(pcd)
            elif form == "voxel":
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
                vis.add_geometry(voxel_grid)
            elif form == "octree":
                octree = o3d.geometry.Octree(max_depth=octree_max_depth)
                octree.convert_from_point_cloud(pcd, size_expand=0.01)
                vis.add_geometry(octree)
            elif form == "empty":
                pass
            else:
                assert False, "form参数错误"

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

            cam_params.intrinsic.set_intrinsics(self.window_width,
                                                self.window_height,
                                                fx=self.window_width * focal,
                                                fy=self.window_width * focal,
                                                cx=self.window_width / 2,
                                                cy=self.window_height / 2)

            vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

        vis.run()
        vis.destroy_window()


    def draw_one_frame(self, scene, frame_id, form="point", point_size=2.0, voxel_size=0.5, octree_max_depth=8,
                        axis=5, filter=None, init_camera_rpy=None, init_camera_T=None):
        '''
        绘制场景的某一帧
        :param scene: 场景加载器
        :param frame_id: 帧id
        :param form: 绘制形式, point表示点云, voxel表示体素化, octree表示八叉树, empty表示不绘制
        :param point_size: 点云大小，当form为point时有效
        :param voxel_size: 体素大小，当form为voxel时有效
        :param octree_max_depth: 八叉树最大深度，当form为octree时有效
        :param axis: 坐标轴大小, None表示不绘制坐标轴
        :param filter: 过滤函数
        :param init_camera_rpy: 相机初始姿态 [roll, pitch, yaw]
        :param init_camera_T: 相机初始位置 [x, y, z]
        :return:
        '''

        pcd_xyz, other_data = scene.get_frame(frame_id=frame_id, filter=filter)

        self.draw_points(pcd_xyz, other_data, form=form, point_size=point_size, voxel_size=voxel_size,
                            octree_max_depth=octree_max_depth, axis=axis, init_camera_rpy=init_camera_rpy,
                            init_camera_T=init_camera_T)



    def play_scene(self, scene, begin=0, end=-1, delay_time=0.1,
                    form="point", point_size=2.0, voxel_size=0.5, octree_max_depth=8,
                    axis=5, filter=None, init_camera_rpy=None, init_camera_T=None):
        '''
        播放场景
        :param scene: 场景加载器
        :param begin: 开始帧
        :param end: 结束帧, -1表示最后一帧
        :param delay_time: 延迟时间
        :param form: 绘制形式, point表示点云, voxel表示体素化, octree表示八叉树, empty表示不绘制
        :param point_size: 点云大小，当form为point时有效
        :param voxel_size: 体素大小，当form为voxel时有效
        :param octree_max_depth: 八叉树最大深度，当form为octree时有效
        :param axis: 坐标轴大小, None表示不绘制坐标轴
        :param filter: 过滤函数
        :param init_camera_rpy: 相机初始姿态 [roll, pitch, yaw]
        :param init_camera_T: 相机初始位置 [x, y, z]
        :return:
        '''
        self.__reset()
        if end == -1:
            end = scene.frame_num - 1

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.window_name, height=self.window_height, width=self.window_width,
                            left=self.window_left, top=self.window_top)

        # 背景颜色
        vis.get_render_option().background_color = np.asarray(self.background_color)

        # 点云大小
        vis.get_render_option().point_size = point_size

        reset_view = False

        pcd = o3d.geometry.PointCloud()

        # 坐标轴
        if axis is not None:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis, origin=[0, 0, 0])

        self.__keybord_callback()

        self.frame_id = begin
        while self.frame_id <= end:
            print("frame_id:", self.frame_id)
            # print("frame_name:", scene.dataset_loader.filenames[self.frame_id])

            # 移除上一帧的几何对象
            vis.clear_geometries()

            # 坐标轴
            if axis is not None:
                vis.add_geometry(ax, reset_bounding_box=False)

            # 点云
            pcd_xyz, other_data = scene.get_frame(frame_id=self.frame_id, filter=filter)
            if pcd_xyz.shape[0] != 0:
                pcd.points = o3d.utility.Vector3dVector(pcd_xyz)

                # 颜色
                if 'pointinfo-color' in other_data.keys():
                    pcd.colors = o3d.utility.Vector3dVector(other_data['pointinfo-color'])

                if form == "point":
                    vis.add_geometry(pcd, reset_bounding_box=False)
                elif form == "voxel":
                    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
                    vis.add_geometry(voxel_grid, reset_bounding_box=False)
                elif form == "octree":
                    octree = o3d.geometry.Octree(max_depth=octree_max_depth)
                    octree.convert_from_point_cloud(pcd, size_expand=0.01)
                    vis.add_geometry(octree, reset_bounding_box=False)
                elif form == "empty":
                    pass
                else:
                    assert False, "form参数错误"

            # 候选框
            if 'geometry-bboxes' in other_data.keys():
                for bbox in other_data['geometry-bboxes']:
                    vis.add_geometry(bbox, reset_bounding_box=False)

            # 箭头
            if 'geometry-arrows' in other_data.keys():
                for arrow in other_data['geometry-arrows']:
                    vis.add_geometry(arrow, reset_bounding_box=False)

            # 球
            if 'geometry-spheres' in other_data.keys():
                for sphere in other_data['geometry-spheres']:
                    vis.add_geometry(sphere, reset_bounding_box=False)

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

    def draw_global_map(self, scene, step=1, axis=5, point_size=2.0):
        '''
        绘制全局地图
        :param scene: 场景加载器
        :param step: 采样步长，每隔step帧采样一次
        :param axis: 坐标轴大小, None表示不绘制坐标轴
        :param point_size: 点云大小
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

        self.draw_points(points_world, axis=axis, point_size=point_size)


    def compare_two_point_clouds(self, pcd_xyz1, pcd_xyz2, other_data1=None, other_data2=None,
                                    form1="point", point_size1=2.0, voxel_size1=0.5, octree_max_depth1=8,
                                    form2="point", point_size2=2.0, voxel_size2=0.5, octree_max_depth2=8,
                                    axis=5, init_camera_rpy=None, init_camera_T=None, camera_sync=True):
        '''
        比较两个点云, 同步视角显示
        :param pcd_xyz1: 点云1
        :param pcd_xyz2: 点云2
        :param other_data1: 其他数据1, 可能包含颜色color, 候选框bbox, 箭头arrows等
        :param other_data2: 其他数据2, 可能包含颜色color, 候选框bbox, 箭头arrows等
        :param form1: 点云1绘制形式, point表示点云, voxel表示体素化, octree表示八叉树, empty表示不绘制
        :param point_size1: 点云1大小，当form1为point时有效
        :param voxel_size1: 点云1体素大小，当form1为voxel时有效
        :param octree_max_depth1: 点云1八叉树最大深度，当form1为octree时有效
        :param form2: 点云2绘制形式, point表示点云, voxel表示体素化, octree表示八叉树, empty表示不绘制
        :param point_size2: 点云2大小，当form2为point时有效
        :param voxel_size2: 点云2体素大小，当form2为voxel时有效
        :param octree_max_depth2: 点云2八叉树最大深度，当form2为octree时有效
        :param axis: 坐标轴大小, None表示不绘制坐标轴
        :param init_camera_rpy: 相机初始姿态 [roll, pitch, yaw]
        :param init_camera_T: 相机初始位置 [x, y, z]
        :param camera_sync: 是否同步相机视角
        :return:
        '''

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
        vis1.get_render_option().point_size = point_size1
        vis2.get_render_option().point_size = point_size2

        # 坐标轴
        if axis is not None:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis, origin=[0, 0, 0])  # 坐标轴
            vis1.add_geometry(ax)
            vis2.add_geometry(ax)

        # 点云
        pcd1 = o3d.geometry.PointCloud()
        if pcd_xyz1.shape[0] != 0:
            pcd1.points = o3d.utility.Vector3dVector(pcd_xyz1)

            # 颜色
            if other_data1 is not None and 'pointinfo-color' in other_data1.keys():
                pcd1.colors = o3d.utility.Vector3dVector(other_data1['pointinfo-color'])

            if form1 == "point":
                vis1.add_geometry(pcd1)
            elif form1 == "voxel":
                voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd1, voxel_size=voxel_size1)
                vis1.add_geometry(voxel_grid1)
            elif form1 == "octree":
                octree1 = o3d.geometry.Octree(max_depth=octree_max_depth1)
                octree1.convert_from_point_cloud(pcd1, size_expand=0.01)
                vis1.add_geometry(octree1)
            elif form1 == "empty":
                pass
            else:
                assert False, "form1参数错误"

        pcd2 = o3d.geometry.PointCloud()
        if pcd_xyz2.shape[0] != 0:
            pcd2.points = o3d.utility.Vector3dVector(pcd_xyz2)

            if other_data2 is not None and 'pointinfo-color' in other_data2.keys():
                pcd2.colors = o3d.utility.Vector3dVector(other_data2['pointinfo-color'])

            if form2 == "point":
                vis2.add_geometry(pcd2)
            elif form2 == "voxel":
                voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd2, voxel_size=voxel_size2)
                vis2.add_geometry(voxel_grid2)
            elif form2 == "octree":
                octree2 = o3d.geometry.Octree(max_depth=octree_max_depth2)
                octree2.convert_from_point_cloud(pcd2, size_expand=0.01)
                vis2.add_geometry(octree2)
            elif form2 == "empty":
                pass
            else:
                assert False, "form2参数错误"

        # 候选框
        if other_data1 is not None and 'geometry-bboxes' in other_data1.keys():
            for bbox in other_data1['geometry-bboxes']:
                vis1.add_geometry(bbox)
        if other_data2 is not None and 'geometry-bboxes' in other_data2.keys():
            for bbox in other_data2['geometry-bboxes']:
                vis2.add_geometry(bbox)

        # 箭头
        if other_data1 is not None and 'geometry-arrows' in other_data1.keys():
            for arrow in other_data1['geometry-arrows']:
                vis1.add_geometry(arrow)
        if other_data2 is not None and 'geometry-arrows' in other_data2.keys():
            for arrow in other_data2['geometry-arrows']:
                vis2.add_geometry(arrow)

        # 球
        if other_data1 is not None and 'geometry-spheres' in other_data1.keys():
            for sphere in other_data1['geometry-spheres']:
                vis1.add_geometry(sphere)
        if other_data2 is not None and 'geometry-spheres' in other_data2.keys():
            for sphere in other_data2['geometry-spheres']:
                vis2.add_geometry(sphere)

        # text = o3d.t.geometry.TriangleMesh.create_text("Hello Open3D", depth=0.1).to_legacy()
        # text.paint_uniform_color((1, 0, 0))
        # vis.add_geometry(text)

        # 初始化视角
        vis1.reset_view_point(True)
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
            if camera_sync:
                cam1 = vis1.get_view_control().convert_to_pinhole_camera_parameters()
                vis2.get_view_control().convert_from_pinhole_camera_parameters(cam1)
            running1 = vis1.poll_events()
            if camera_sync:
                cam2 = vis2.get_view_control().convert_to_pinhole_camera_parameters()
                vis1.get_view_control().convert_from_pinhole_camera_parameters(cam2)
            running2 = vis2.poll_events()

            vis1.update_renderer()
            vis2.update_renderer()

        vis1.destroy_window()
        vis2.destroy_window()


    def compare_one_frame(self, scene, frame_id, axis=5, filter1=None, filter2=None,
                            form1="point", point_size1=2.0, voxel_size1=0.5, octree_max_depth1=8,
                            form2="point", point_size2=2.0, voxel_size2=0.5, octree_max_depth2=8,
                            init_camera_rpy=None, init_camera_T=None, camera_sync=True):
        '''
        比较两个过滤器的结果，同步视角显示
        :param scene: 场景加载器
        :param frame_id: 帧id
        :param axis: 坐标轴大小, None表示不绘制坐标轴
        :param filter1: 过滤函数1
        :param filter2: 过滤函数2
        :param form1: 点云1绘制形式, point表示点云, voxel表示体素化, octree表示八叉树, empty表示不绘制
        :param point_size1: 点云1大小，当form1为point时有效
        :param voxel_size1: 点云1体素大小，当form1为voxel时有效
        :param octree_max_depth1: 点云1八叉树最大深度，当form1为octree时有效
        :param form2: 点云2绘制形式, point表示点云, voxel表示体素化, octree表示八叉树, empty表示不绘制
        :param point_size2: 点云2大小，当form2为point时有效
        :param voxel_size2: 点云2体素大小，当form2为voxel时有效
        :param octree_max_depth2: 点云2八叉树最大深度，当form2为octree时有效
        :param init_camera_rpy: 相机初始姿态 [roll, pitch, yaw]
        :param init_camera_T: 相机初始位置 [x, y, z]
        :param camera_sync: 是否同步相机视角
        :return:
        '''

        pcd_xyz1, other_data1 = scene.get_frame(frame_id=frame_id, filter=filter1)
        pcd_xyz2, other_data2 = scene.get_frame(frame_id=frame_id, filter=filter2)

        self.compare_two_point_clouds(pcd_xyz1, pcd_xyz2, other_data1, other_data2,
                                        form1=form1, point_size1=point_size1, voxel_size1=voxel_size1, octree_max_depth1=octree_max_depth1,
                                        form2=form2, point_size2=point_size2, voxel_size2=voxel_size2, octree_max_depth2=octree_max_depth2,
                                        axis=axis, init_camera_rpy=init_camera_rpy, init_camera_T=init_camera_T, camera_sync=camera_sync)


    def compare_scene(self, scene, filter1=None, filter2=None, delay_time=0.1, begin=0, end=-1,
                        form1="point", point_size1=2.0, voxel_size1=0.5, octree_max_depth1=8,
                        form2="point", point_size2=2.0, voxel_size2=0.5, octree_max_depth2=8,
                        axis=5, init_camera_rpy=None, init_camera_T=None, camera_sync=True):
        '''
        播放场景并比较两个过滤函数的结果, 默认开启同步视角显示
        :param scene: 场景加载器
        :param filter1: 过滤函数1
        :param filter2: 过滤函数2
        :param delay_time: 延迟时间
        :param begin: 开始帧, 从0开始
        :param end: 结束帧, -1表示最后一帧
        :param form1: 点云1绘制形式, point表示点云, voxel表示体素化, octree表示八叉树, empty表示不绘制
        :param point_size1: 点云1大小，当form1为point时有效
        :param voxel_size1: 点云1体素大小，当form1为voxel时有效
        :param octree_max_depth1: 点云1八叉树最大深度，当form1为octree时有效
        :param form2: 点云2绘制形式, point表示点云, voxel表示体素化, octree表示八叉树, empty表示不绘制
        :param point_size2: 点云2大小，当form2为point时有效
        :param voxel_size2: 点云2体素大小，当form2为voxel时有效
        :param octree_max_depth2: 点云2八叉树最大深度，当form2为octree时有效
        :param axis: 坐标轴大小, None表示不绘制坐标轴
        :param init_camera_rpy: 相机初始姿态 [roll, pitch, yaw]
        :param init_camera_T: 相机初始位置 [x, y, z]
        :param camera_sync: 是否同步相机视角
        :return:
        '''
        self.__reset()
        if end == -1:
            end = scene.frame_num - 1

        reset_view = False

        # 第一个窗口
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
        vis1.get_render_option().point_size = point_size1
        vis2.get_render_option().point_size = point_size2

        # 点云
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()

        # 坐标轴
        if axis is not None:
            ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis, origin=[0, 0, 0])

        self.__keybord_callback()

        self.frame_id = begin
        while self.frame_id <= end:
            # print("frame_id:", self.frame_id)
            # print("frame_name:", scene.dataset_loader.filenames[self.frame_id])

            # # 移除上一帧的几何对象
            vis1.clear_geometries()
            vis2.clear_geometries()

            # 坐标轴
            if axis is not None:
                vis1.add_geometry(ax, reset_bounding_box=False)
                vis2.add_geometry(ax, reset_bounding_box=False)

            # 点云
            pcd_xyz1, other_data1 = scene.get_frame(frame_id=self.frame_id, filter=filter1)
            if pcd_xyz1.shape[0] != 0:
                pcd1.points = o3d.utility.Vector3dVector(pcd_xyz1)

                # 颜色
                if 'pointinfo-color' in other_data1.keys():
                    pcd1.colors = o3d.utility.Vector3dVector(other_data1['pointinfo-color'])

                if form1 == "point":
                    vis1.add_geometry(pcd1, reset_bounding_box=False)
                elif form1 == "voxel":
                    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd1, voxel_size=voxel_size1)
                    vis1.add_geometry(voxel_grid1, reset_bounding_box=False)
                elif form1 == "octree":
                    octree1 = o3d.geometry.Octree(max_depth=octree_max_depth1)
                    octree1.convert_from_point_cloud(pcd1, size_expand=0.01)
                    vis1.add_geometry(octree1, reset_bounding_box=False)
                elif form1 == "empty":
                    pass
                else:
                    assert False, "form1参数错误"


            pcd_xyz2, other_data2 = scene.get_frame(frame_id=self.frame_id, filter=filter2)
            if pcd_xyz2.shape[0] != 0:
                pcd2.points = o3d.utility.Vector3dVector(pcd_xyz2)

                if 'pointinfo-color' in other_data2.keys():
                    pcd2.colors = o3d.utility.Vector3dVector(other_data2['pointinfo-color'])

                if form2 == "point":
                    vis2.add_geometry(pcd2, reset_bounding_box=False)
                elif form2 == "voxel":
                    voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd2, voxel_size=voxel_size2)
                    vis2.add_geometry(voxel_grid2, reset_bounding_box=False)
                elif form2 == "octree":
                    octree2 = o3d.geometry.Octree(max_depth=octree_max_depth2)
                    octree2.convert_from_point_cloud(pcd2, size_expand=0.01)
                    vis2.add_geometry(octree2, reset_bounding_box=False)
                elif form2 == "empty":
                    pass
                else:
                    assert False, "form2参数错误"

            # 候选框
            if 'geometry-bboxes' in other_data1.keys():
                for bbox in other_data1['geometry-bboxes']:
                    vis1.add_geometry(bbox, reset_bounding_box=False)
            if 'geometry-bboxes' in other_data2.keys():
                for bbox in other_data2['geometry-bboxes']:
                    vis2.add_geometry(bbox, reset_bounding_box=False)

            # 箭头
            if 'geometry-arrows' in other_data1.keys():
                for arrow in other_data1['geometry-arrows']:
                    vis1.add_geometry(arrow, reset_bounding_box=False)
            if 'geometry-arrows' in other_data2.keys():
                for arrow in other_data2['geometry-arrows']:
                    vis2.add_geometry(arrow, reset_bounding_box=False)

            # 球
            if 'geometry-spheres' in other_data1.keys():
                for sphere in other_data1['geometry-spheres']:
                    vis1.add_geometry(sphere, reset_bounding_box=False)
            if 'geometry-spheres' in other_data2.keys():
                for sphere in other_data2['geometry-spheres']:
                    vis2.add_geometry(sphere, reset_bounding_box=False)

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
                    cam_params.intrinsic.set_intrinsics(self.window_width,
                                                        self.window_height,
                                                        fx=self.window_width * focal,
                                                        fy=self.window_width * focal,
                                                        cx=self.window_width / 2,
                                                        cy=self.window_height / 2)

                    vis1.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

                reset_view = True

            if camera_sync:
                cam1 = vis1.get_view_control().convert_to_pinhole_camera_parameters()
                vis2.get_view_control().convert_from_pinhole_camera_parameters(cam1)
            running1 = vis1.poll_events()
            if camera_sync:
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
                if camera_sync:
                    cam1 = vis1.get_view_control().convert_to_pinhole_camera_parameters()
                    vis2.get_view_control().convert_from_pinhole_camera_parameters(cam1)
                running1 = vis1.poll_events()
                if camera_sync:
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
                if camera_sync:
                    cam1 = vis1.get_view_control().convert_to_pinhole_camera_parameters()
                    vis2.get_view_control().convert_from_pinhole_camera_parameters(cam1)
                running1 = vis1.poll_events()
                if camera_sync:
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

