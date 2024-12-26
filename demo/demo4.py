# -*- coding: utf-8 -*-
"""
@Time        :  2024/6/5 14:00
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  demo4 演示4, Filters和Tools的综合使用，构建由多个过滤器组成的自定义过滤函数
"""
import sys
sys.path.append('.')
from options import options  # 导入参数设置
from sceneloader import SceneLoader  # 导入场景加载器
from utils import Visualizer, Tools, Filters

def filter(pcd_xyz, other_data):
    '''
    过滤函数
    :param pcd_xyz: 点云 [N, 3]
    :param other_data: 其他数据
    :return: pcd_xyz, other_data
    '''
    pcd_xyz, other_data = Filters.add_noise_v(pcd_xyz, other_data)  # 为速度添加高斯噪声

    pcd_xyz, other_data = Filters.add_noise_xyz(pcd_xyz, other_data)  # 在3d点云射线长度上添加高斯噪声

    id_times = Tools.get_id_times(other_data['pointinfo-id'])  # 获取每个id的出现次数

    # 移除出现次数前2的id
    remove_id = id_times[:2, 0].tolist()
    print('remove id:', remove_id)

    pcd_xyz, other_data = Filters.remove_points_by_id(pcd_xyz, other_data, id_list=remove_id)  # 移除指定id的点云

    return pcd_xyz, other_data


if __name__ == '__main__':
    opt = options()
    opt.dataset = 'carla_4d'
    opt.scene_id = 0

    visualizer = Visualizer(opt)

    # 加载场景
    scene = SceneLoader(opt)
    print(scene.frame_num)

    # 播放场景
    visualizer.play_scene(scene, filter=filter, begin=0, delay_time=0)

