# -*- coding: utf-8 -*-
from core import load_config, SceneLoader, Visualizer, FrameData
import numpy as np

# 运行方式： python -m demos.demo2_filter

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

    # 加载场景和可视化工具
    scene = SceneLoader(config.scene_config)
    visualizer = Visualizer(config.visualizer_config)
    
    # -----------------------------------------------------------------------------------------
    # 方式1：使用scene.get_frame()获取帧数据后，手动调用过滤函数进行处理，然后通过visualizer.draw_points()可视化处理后的帧数据
    frame_data = scene.get_frame(frame_id=100)  # 获取第100帧点云
    frame_data = remove_specified_id(frame_data)  # 使用过滤函数处理帧数据
    visualizer.draw_points(frame_data)  # 可视化处理后的帧数据

    # -----------------------------------------------------------------------------------------
    # 方式2：在调用scene.get_frame()时直接传入过滤函数，获取经过过滤处理后的帧数据，然后通过visualizer.draw_points()可视化
    frame_data = scene.get_frame(frame_id=100, filter=remove_specified_id)
    visualizer.draw_points(frame_data)

    # -----------------------------------------------------------------------------------------
    # 方式3（推荐）：直接调用visualizer.draw_one_frame()，在其中传入过滤函数，绘制经过过滤处理后的单帧数据
    visualizer.draw_one_frame(scene, frame_id=100, filter=remove_specified_id)  # 绘制一帧，使用过滤函数

    # -----------------------------------------------------------------------------------------
    # 方式4（推荐）：直接调用visualizer.play_scene()，在其中传入过滤函数，动态播放整个场景，播放过程中每一帧都会经过过滤函数处理
    # 可以使用 空格键 暂停/继续，在暂停状态下可以使用方向键 ← → 或 ↑ ↓ 来控制帧的前进和后退
    visualizer.play_scene(scene, frame_range=[100, -1], filter=remove_specified_id)
