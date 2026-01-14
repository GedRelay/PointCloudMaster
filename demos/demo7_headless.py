# -*- coding: utf-8 -*-
from core import load_config, SceneLoader, Visualizer, Tools
import time
import matplotlib.pyplot as plt

times = []

def filter(frame_data):
    start_time = time.time()
    labels = Tools.Clustering.kmeans(frame_data.pcd.points, n_clusters=10)
    end_time = time.time()
    times.append(end_time - start_time)
    print("time consumption for k-means clustering:", end_time - start_time, "seconds")
    return frame_data


if __name__ == '__main__':
    config = load_config('core/default_config.yaml')
    config.scene_config.dataset = 'aeva'
    config.scene_config.scene_id = 0

    scene = SceneLoader(config.scene_config)
    visualizer = Visualizer(config.visualizer_config)

    # 无头模式下不对数据进行可视化，用以遍历场景对算法快速验证
    visualizer.headless(scene, filter=filter, frame_range=[0, 5])
    
    # 绘制时间消耗的曲线
    plt.plot(range(len(times)), times, marker='o')
    plt.xlabel('Frame Index')
    plt.ylabel('Time (seconds)')
    plt.title('Time Consumption for K-means Clustering')
    plt.grid()
    plt.show()