# -*- coding: utf-8 -*-
"""
@Time        :  2024/6/5 12:44
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  demo0, 演示0，可视化点云
"""

import sys
sys.path.append('.')
from options import Options  # 导入参数设置
from utils.visualizer import Visualizer  # 导入可视化工具
from utils.sceneloader import SceneLoader  # 导入场景加载工具


if __name__ == '__main__':
    # 设置参数, 也可以在命令行中设置，或者使用options.py的默认参数
    opt = Options().parse()
    opt.dataset = 'carla1'
    opt.scene_id = 0
    opt.preload = True  # 预加载
    opt.preload_begin = 0
    opt.preload_end = 100

    # 加载场景
    scene = SceneLoader(opt)
    print("场景帧数:", scene.frame_num)

    # 获取第100帧点云
    pcd_xyz, _ = scene.get_frame(frame_id=100)
    print(pcd_xyz.shape)

    # 创建可视化工具
    visualizer = Visualizer(opt)

    # 可视化点云
    visualizer.draw_points(pcd_xyz)

    # 动态可视化整个场景
    # 可以使用 空格键 暂停/继续，在暂停状态下可以使用方向键 ← → 或 ↑ ↓ 来控制帧的前进和后退
    visualizer.play_scene(scene, delay_time=0)
