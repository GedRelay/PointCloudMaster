# -*- coding: utf-8 -*-
"""
@Time        :  2024/6/5 上午1:42
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  demo3, 演示3, 查看全局地图
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

from options import Options
from utils.visualizer import Visualizer
from utils.sceneloader import SceneLoader

if __name__ == '__main__':
    opt = Options().parse()

    # 设置参数
    opt.dataset = 'carla2'
    opt.scene_id = 0

    # 加载场景
    scene = SceneLoader(opt)
    print("总帧数：", scene.frame_num)

    # 查看全局地图
    visualizer = Visualizer(opt)
    visualizer.draw_global_map(scene, step=20)