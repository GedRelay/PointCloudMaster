# PointCloudMaster

这是一个室外点云可视化的代码框架，通过编写自己的"过滤函数"实现对点云的处理，并能方便地对处理后的数据进行可视化



# 目录结构说明

本项目文件目录如下

```
PointCloudMaster
├─demo
│  ├─demo0.py
│  ├─demo1.py
│  ├─demo2.py
│  ├─demo3.py
│  ├─demo4.py
│  ├─demo5.py
│  └─demo6.py
├─utils
│  ├─__init__.py
│  ├─filters.py
│  ├─tools.py
│  └─visualizer.py
├─sceneloader
│  ├─__init__.py
│  └─scene_loader.py
├─options.py
├─requirements.txt
└─README.md
```

- `demo`：帮助你快速上手的演示示例
- `utils`：包含可视化工具以及其他工具函数
- `sceneloader`：数据加载相关文件
- `options.py`：默认运行参数
- `requirements.txt`：项目依赖包列表
- `README.md`：就是你现在在看的这个文档



# 快速上手

安装：

1. 作者的python版本为3.8.18, 请确保你使用的环境中python版本为3.8

2. 运行`pip install -r requirements.txt` 指令安装依赖

3. 请阅读并运行`demo`文件夹下的演示代码



# Tools类

其中包含许多工具函数

| 函数                     | 作用                       |
| ---------------------- | ------------------------ |
| `get_bbox_from_points` | 从点云中获取包围盒                |
| `get_arrow`            | 获取箭头                     |
| `euler2mat`            | 欧拉角转旋转矩阵                 |
| `get_id_times`         | 获取每个id的出现次数              |
| `get_sphere`           | 获取球体                     |
| `dbscan`               | 使用sklearn的dbscan进行聚类     |
| `dbscan2`              | 使用open3d的dbscan进行聚类      |
| `mean_shift`           | 使用sklearn的mean_shift进行聚类 |
| `kmeans`               | 使用sklearn的kmeans进行聚类     |
| `xyz2abrho`            | 将三维空间点云转换为极坐标空间点云        |


# Filters类

其中包含很多常用的过滤函数


| 函数                        | 作用               |
| :------------------------ | :--------------- |
| `xyz2v`                   | 将三维空间点云转换为速度空间点云 |
| `add_noise_v`             | 为速度添加高斯噪声        |
| `add_noise_xyz`           | 在3d点云射线长度上添加高斯噪声 |
| `remove_points_by_mask`   | 去除mask所对应的点      |
| `remain_points_by_mask`   | 保留mask所对应的点      |
| `remove_points_by_id`     | 通过id去除点          |
| `remain_points_by_id`     | 通过id保留点          |
| `remain_points_by_z_axis` | 保留z轴在一定范围之间的点    |




# 如何添加自定义数据集

1. 在`sceneloader/datasets.json`中按照格式添加数据集每个场景的点云存放目录以及位姿文件位置

```json
{
  "datasets": 
  [
    ...其他数据集

    {
      "name": "这里填写数据集的名字，不要与其他数据集相同",
      "root_path": "数据集根目录位置",
      "scenes": [
        {
          "scene_id": 0,
          "pcd_path": "0号场景点云存放目录，确保该目录下只有点云文件",
          "pose_path": "0号场景位姿文件位置，如果没有则填写null", 
          "vehicle_state_path": "0号场景车辆状态文件位置，如果没有则填写null"
        },
        {
          "scene_id": 1,
          "pcd_path": "1号场景点云存放目录，确保该目录下只有点云文件",
          "pose_path": null,
          "vehicle_state_path": null
        },
        ...
      ]
    }

  ]
}
```

2. 在`sceneloader`目录下添加python文件，文件命名为：`数据集名字.py`
    1. 文件中按照以下模板进行实现，注意：类的命名要为`数据集名字`
    2.  `load_frame(self, frame_id)`：加载某一帧的数据。返回一个大小为 `N*3` 的 `numpy` 点云数据`pcd_xyz`，和一个字典`other_data`，该字典的key为字符串，值为对应数据。每个数据集包含的数据都有区别，这些数据都被存放在`other_data`中
    3.  `load_poses(self, scene_id)`：加载所有帧的位姿。返回旋转矩阵的列表`Rs`和平移向量的列表`Ts`，`Rs`中每个元素为大小为`3*3`的`numpy`数组，`Ts`中每个元素为长度为`3`的`numpy`数组。如果该数据集没有位姿信息，则不需要实现该函数
    4. `load_vehicle_state(self, scene_id)`：加载所有帧的车辆状态信息。返回一个字典，该字典的key为字符串，值为对应数据

```python
from sceneloader import DatasetLoader_Base
import os
import numpy as np

class name(DatasetLoader_Base):
    def __init__(self, scene_id, json_data):
        super(DatasetLoader_Name, self).__init__(scene_id, json_data)

    def load_frame(self, frame_id):
        pcd_path = os.path.join(self.pcd_data_path, self.filenames[frame_id])
        # 在此实现方法以读取点云
        ...

    def load_poses(self, scene_id):
        # 在此实现方法以读取位姿，如果没有位姿数据则可以不实现
```



## other_data说明

`other_data`是点云数据中除了`x,y,z`以外的所有数据。由于不同数据集的点云字段各不相同，因此设计了该数据结构

`other_data`是一个字典，键为字符串，值为任意其他数据。注意，字符串的命名根据其作用是具有一定的要求的：

- 以`pointinfo-`开头的数据表示点云中每个点的属性值，比如颜色`pointinfo-color`，标签`pointinfo-label`等。这些数据的长度要与点云数量对应。其中`pointinfo-color`较为特殊，若有，则会在可视化中绘制出来
- 以`geometry-`开头的数据表示几何元素，若有，则会在可视化中绘制出来。目前支持的有`geometry-bboxes`, `geometry-arrows`, `geometry-spheres`。这些数据为列表类型，列表中为单个的几何元素
- 以`pose-`开头的数据表示位姿信息，`pose-R`表示旋转矩阵，`pose-T`表示平移向量，由load_poses函数读取
- 以`vehicle-`开头的数据表示自车信息，如`vehicle-real_v`表示车辆速度，`vehicle-acc`表示车辆加速度，由load_vehicle_state函数读取


# 数据集说明

## carla1

有 2 个场景

每一帧的每一个点云的数据格式为
```
x, y, z, rv, vx, vy, vz, id, label, intensity
```
- `x, y, z`: 点云空间位置
- `rv`: 径向速度 (**未补偿**)
- `vx, vy, vz`: 真实矢量速度
- `id`: 实体 id，静态物体的 id 为 `98` 
- `label`: 语义标注（**全是 `0`，可以认为该数据集没有语义标签**）
- `intensity`: 强度（没有使用过，不清楚有没有问题）


>**注意！**
>1. 该数据集没有车辆的位姿信息
>2. 标注的真实速度 `vx, vy, vz` 是以全局地图为参考系下的速度，如果拥有车辆的位姿信息的话可以将全局地图参考系下的速度转换为主车参考系下的速度，正如第一条所说该数据集没有车辆的位姿信息
>3. 语义标注 `label` 列全都是 `0`，可以认为该数据集没有语义标签




## carla2
有 1 个场景

每一帧的每一个点云的数据格式为
```
x, y, z, cosangle, rv, vx, vy, vz, vcps, id, label, intensity
```
- `x, y, z`: 点云空间位置
- `cosangle`: 
- `rv`: 径向速度 (**未补偿**)
- `vx, vy, xz`: 真实矢量速度 (**全是 `0`,可以认为该数据集没有真实速度标签** )
- `vcps`: 补偿后的径向速度
- `id`: 实体 id
- `label`: 语义标签
- `intensity`: 强度（没有使用过，不清楚有没有问题）


>**注意！** 
>1. 真实速度标签 `vx,vy,vz` 列全部为 `0`，可以认为该数据集没有真实速度标签



## aeva
有 8 个场景

每一帧的每一个点云的数据格式为
```
x, y, z, rv, time
```
- `x, y, z`: 点云空间位置
- `rv`: 径向速度（**未补偿**）
- `time`: 被扫描到的时间




## helipr
有 1 个场景

每一帧的每一个点云的数据格式为
```
x, y, z, reflectivity, vcps, time, line_index, intensity
```
- `x, y, z`: 点云空间位置
- `reflectivity`: 反射率
- `vcps`: 补偿后的径向速度
- `time`: 被扫描到的时间
- `line_index`: 扫描线索引
- `intensity`: 强度


其他信息：\
一共有 64 条扫描线（0 到 63 号从最下面到最上面）\
但这几条线不是按顺序扫描的（通过对 time 进行排序发现）\
但大致是从左到右，从下到上的扫描顺序

两帧之间间隔不到 1 秒，约 10 帧每秒