# -*- coding: utf-8 -*-
from core import load_config, SceneLoader, Visualizer

if __name__ == '__main__':
    config = load_config('core/default_config.yaml')
    config.scene_config.dataset = 'carla4d'
    config.scene_config.scene_id = 0

    # 加载场景
    scene = SceneLoader(config.scene_config)
    print("总帧数：", scene.frame_num)

    # 查看全局地图
    visualizer = Visualizer(config.visualizer_config)
    visualizer.draw_global_map(scene, step=40)