# -*- coding: utf-8 -*-
"""
@Time        :  2024/6/4 18:14
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  options 参数设置
"""
import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        '''
        初始化参数
        :return:
        '''
        # 场景加载参数
        self.parser.add_argument('--dataset', type=str, default='carla1', help='数据集名称: carla1, carla2, aeva, helipr')
        self.parser.add_argument('--scene_id', type=int, default=0, help='场景编号, 从0开始')
        self.parser.add_argument('--preload', type=bool, default=False, help='是否预加载数据')
        self.parser.add_argument('--preload_begin', type=int, default=0, help='预加载数据的起始帧')
        self.parser.add_argument('--preload_end', type=int, default=-1, help='预加载数据的结束帧, -1表示到最后一帧')

        # 窗口参数
        self.parser.add_argument('--window_name', type=str, default='Visualizer', help='窗口名称')
        self.parser.add_argument('--window_height', type=int, default=540, help='窗口高度')
        self.parser.add_argument('--window_width', type=int, default=960, help='窗口宽度')
        self.parser.add_argument('--window_left', type=int, default=50, help='窗口生成时的左边距')
        self.parser.add_argument('--window_top', type=int, default=50, help='窗口生成时的上边距')

        # 渲染参数
        self.parser.add_argument('--background_color', type=list, default=[1, 1, 1], help='背景颜色, 单值范围为[0, 1]')

    def parse(self):
        '''
        解析参数
        :return:
        '''
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt