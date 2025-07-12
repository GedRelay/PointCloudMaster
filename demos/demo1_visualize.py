from core import load_config, SceneLoader, Visualizer


# 运行方式： python -m demos.demo1_visualize
if __name__ == '__main__':
    # 加载参数和设置:
    # 1. 可以手动修改目标配置文件*.yaml中的参数
    # 2. 也可以通过命令行参数传入配置进行覆盖， 例如：python -m demos.demo1_visualize --scene_config.dataset carla_4d --scene_config.scene_id 0
    # 3. 也可以在代码中直接修改配置进行覆盖（如下）
    config = load_config('core/default_config.yaml')
    config.scene_config.dataset = 'carla4d'
    config.scene_config.scene_id = 0

    # 加载场景
    scene = SceneLoader(config.scene_config)
    print("场景帧数:", scene.frame_num)

    # 获取第100帧点云
    frame_data = scene.get_frame(frame_id=100)
    print(frame_data.pcd.points.shape)

    # 创建可视化工具
    visualizer = Visualizer(config.visualizer_config)

    # 可视化单帧点云
    visualizer.draw_points(frame_data)

    # 动态可视化整个场景
    # 可以使用 空格键 暂停/继续，在暂停状态下可以使用方向键 ← → 或 ↑ ↓ 来控制帧的前进和后退
    visualizer.play_scene(scene)
