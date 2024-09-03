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
        # 自定义参数
        self.parser.add_argument('--dataset', type=str, default='carla1', help='数据集名称: carla1, carla2, aeva, helipr')
        self.parser.add_argument('--scene_id', type=int, default=0, help='场景编号, 从0开始')

        # 窗口参数
        self.parser.add_argument('--window_name', type=str, default='Visualizer', help='窗口名称')
        self.parser.add_argument('--window_height', type=int, default=540, help='窗口高度')
        self.parser.add_argument('--window_width', type=int, default=960, help='窗口宽度')
        self.parser.add_argument('--window_left', type=int, default=50, help='窗口生成时的左边距')
        self.parser.add_argument('--window_top', type=int, default=50, help='窗口生成时的上边距')

        # 渲染参数
        self.parser.add_argument('--background_color', type=list, default=[1, 1, 1], help='背景颜色, 单值范围为[0, 1]')
        self.parser.add_argument('--point_size', type=float, default=4.0, help='显示点云大小')

    def parse(self):
        '''
        解析参数
        :return:
        '''
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt