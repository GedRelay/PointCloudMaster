# -*- coding: utf-8 -*-
from core import load_config, SceneLoader, Visualizer, FrameData
import numpy as np


# 定义过滤函数，该函数的输入是帧数据，输出是经过自定义处理后的数据，类型为FrameData
def remove_specified_id(frame_data: FrameData) -> FrameData:
    '''
    过滤函数, 去除指定id的点云数据
    :param frame_data: 帧数据
    '''
    print(frame_data)  # 可以打印帧数据查看其内容

    unique_id = np.unique(frame_data.pcd.id)
    print("当前帧点云的唯一ID:", unique_id)

    mask = (frame_data.pcd.id != 98)

    # 将frame_data.pcd下的所有数据根据mask进行过滤
    for key in frame_data.pcd.keys():
        data = frame_data.pcd[key]
        if data is not None:
            frame_data.pcd[key] = data[mask]

    return frame_data


if __name__ == '__main__':
    # 加载参数和设置
    config = load_config('core/default_config.yaml')
    config.scene_config.dataset = 'carla4d'
    config.scene_config.scene_id = 0
    visualizer = Visualizer(config.visualizer_config)

    # 加载场景
    scene = SceneLoader(config.scene_config)

    # 方式1：获取第100帧经过过滤函数后的帧数据，并绘制
    frame_data = scene.get_frame(frame_id=100, filter=remove_specified_id)
    visualizer.draw_points(frame_data)

    # 方式2：直接调用visualizer.draw_one_frame函数进行绘制
    visualizer.draw_one_frame(scene, frame_id=100, filter=remove_specified_id)  # 绘制一帧，使用过滤函数

    # 播放场景，对每一帧都使用过滤函数进行处理
    visualizer.play_scene(scene, filter=remove_specified_id)
