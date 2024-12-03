# -*- coding: utf-8 -*-
"""
@Time        :  2024/6/5 上午1:42
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  demo3, 演示3, 查看全局地图
"""
import sys
sys.path.append('.')
from options import options  # 导入参数设置
from sceneloader import SceneLoader  # 导入场景加载器
from utils import Visualizer  # 导入可视化工具

if __name__ == '__main__':
    opt = options()

    # 设置参数
    opt.dataset = 'carla3'
    opt.scene_id = 0

    # 加载场景
    scene = SceneLoader(opt)
    print("总帧数：", scene.frame_num)

    # 查看全局地图
    visualizer = Visualizer(opt)
    visualizer.draw_global_map(scene, step=20)