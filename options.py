# -*- coding: utf-8 -*-
"""
@Time        :  2024/6/4 18:14
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  options 参数设置
"""
import argparse

def options():
    '''
    初始化参数
    :return:
    '''
    parser = argparse.ArgumentParser(description='options')
    # 场景加载参数
    parser.add_argument('--jsonfile', type=str, default='datasets.json', help='数据集配置文件')
    parser.add_argument('--dataset', type=str, default='carla_4d', help='数据集名称，请与sceneloader/datasets.json中的名称一致')
    parser.add_argument('--scene_id', type=int, default=0, help='场景编号, 从0开始')
    parser.add_argument('--preload', type=bool, default=False, help='是否预加载数据')
    parser.add_argument('--preload_begin', type=int, default=0, help='预加载数据的起始帧')
    parser.add_argument('--preload_end', type=int, default=-1, help='预加载数据的结束帧, -1表示到最后一帧')

    # 窗口参数
    parser.add_argument('--window_name', type=str, default='Visualizer', help='窗口名称')
    parser.add_argument('--window_height', type=int, default=540, help='窗口高度')
    parser.add_argument('--window_width', type=int, default=960, help='窗口宽度')
    parser.add_argument('--window_left', type=int, default=50, help='窗口生成时的左边距')
    parser.add_argument('--window_top', type=int, default=50, help='窗口生成时的上边距')

    # 渲染参数
    parser.add_argument('--background_color', type=list, default=[1, 1, 1], help='背景颜色, 单值范围为[0, 1]')

    opt = parser.parse_args()
    return opt
