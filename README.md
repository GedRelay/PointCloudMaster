# PointCloudMaster

这是一个室外点云可视化的代码框架，通过编写自己的"过滤函数"实现对点云的处理，并能方便地对处理后的数据进行可视化



# 目录结构说明

本项目主要文件目录如下

```
PointCloudMaster
├─core                              # 存放核心代码
│  ├─config_loader.py               # 配置加载器
│  ├─default_config.yaml            # 默认配置文件
│  ├─filters.py                     # 常用的过滤函数
│  ├─frame_data.py                  # 帧数据类，存储当前帧的所有信息
│  ├─scene_loader.py                # 场景加载器
│  ├─tools.py                       # 常用的工具函数
│  └─visualizer.py                  # 可视化器
├─datasets                          # 存放数据集相关文件
│  ├─datasets.yaml                  # 所有数据集配置文件
│  └─*.py                           # 自定义的特定数据集对象
├─demos                             # 存放演示代码
│  ├─demo1_visualize.py             # 演示1: 基础的可视化演示
│  ├─demo2_filter.py                # 演示2: 过滤函数的使用方法
│  ├─demo3_Tools_and_Filters.py     # 演示3: 自带的工具函数和过滤函数的使用方法
│  ├─demo4_global_map.py            # 演示4: 全局地图的可视化
│  ├─demo5_compare.py               # 演示5: 双窗口对比可视化
│  ├─demo6_form.py                  # 演示6: 点云，体素，八叉树的可视化
│  ├─demo7_kitti_tracking.py        # 演示7: KITTI跟踪数据的可视化
│  └─demo8_headless.py              # 演示8: 无头模式的使用
├─requirements.txt                  # 项目依赖包列表
└─README.md                         # 项目说明文档
```


# 快速上手

安装：

1. 作者的python版本为3.8.18, 不保证其他python版本环境运行正常

2. 运行`pip install -r requirements.txt` 指令安装依赖

3. 修改`datasets/datasets.yaml`文件，添加远程服务器和数据集的相关信息。可以参考已有的数据集配置
```yaml
# datasets/datasets.yaml 文件内容
hosts:
  - hostname: 服务器名字
    ip: 服务器ip
    username: 服务器用户名
    private_key: 本机存放ssh私钥的位置，如 C:/Users/Admin/.ssh/id_rsa
```

4. 请阅读并运行`demo`文件夹下的演示代码，运行以下指令查看演示1的可视化效果，其他演示代码的运行方法类似
```bash
python -m demos.demo1_visualize
```



# 如何添加自定义数据集

1. 在`datasets/datasets.yaml`中按照格式添加数据集每个场景的相关数据文件存放位置
- 如果数据集存放在服务器上，则需要填写`hostname`字段，否则不需要或者填写`localhost`。`hostname`字段与`hosts`设置中的服务器名字对应
- 在`scenes`下的字段中，如果字段名以`path`结尾，将会在后续的代码中被自动拼接上数据集根目录位置形成完整路径，并自动添加到后续的数据集加载器类中，比如可以直接使用`self.pcd_path`来获取点云数据的完整路径


```yaml
hosts:
  - hostname: 服务器名字
    ip: 服务器ip
    username: 服务器用户名
    private_key: 本机存放ssh私钥的位置，如 C:/Users/Admin/.ssh/id_rsa

  ...其他服务器信息

datasets:
  ...其他数据集
  
  - name: 这里填写数据集的名字，不要与其他数据集相同
    hostname: 服务器名字，如果是本地数据集则不需要此字段，或者填写localhost
    root_path: 数据集根目录位置
    scenes:
      - scene_id: 0
        pcd_path: 0号场景点云存放目录，确保该目录下只有点云文件，路径相对于数据集根目录
        后面可以根据数据集继续添加数据集特有的数据路径，如图片路径，标签路径，车辆状态路径等等
        注意：这里以path结尾的字段将会在后续的代码中被自动拼接上数据集根目录位置形成完整路径，并自动添加到后续的数据集加载器中
      - scene_id: 1
        pcd_path: 0号场景点云存放目录，确保该目录下只有点云文件，路径相对于数据集根目录
        ...
  
  ...其他数据集
```


2. 在`datasets`目录下添加python文件，文件命名为：`数据集名字.py`
- 文件中按照以下模板进行实现，注意：类的命名要为`数据集名字`
- `load_frame(self, frame_id)`：加载某一帧的数据，保存到`self.frame_data`中。最后返回`self.frame_data`


```python
from sceneloader import DatasetBase
import os

class name(DatasetBase):
    def __init__(self, scene_id, dataset_config):
        super().__init__(scene_id, dataset_config)
        self.pcd_filenames = self.remote.listdir(self.pcd_path)  # 获取点云文件名列表
        self.pcd_filenames.sort(key=lambda x: int(x.split('.')[0]))  # 假设文件名为数字，按数字排序
        # 在此之后可以添加数据集特有的初始化操作

    def load_frame(self, frame_id):
        pcd_path = os.path.join(self.pcd_path, self.pcd_filenames[frame_id])
        # 在此实现方法以读取点云等数据保存到self.frame_data中
        # 除了点云数据外，还可以读取其他数据，如图片，标注，这些数据都可以存放在self.frame_data中
        ...
        return self.frame_data
```
- 注意，如果数据集存放在服务器上，则需要使用以下方法来获取数据（本地数据这么写也不会有影响）。其本质是将数据从服务器上下载到本地，然后再读取。当with语句结束时，会自动删除本地数据
```python
with self.remote.get(远程数据路径) as 本地数据路径:
    load(本地数据路径)
```
- 注意，如果数据集存放在服务器上，需要获取某个目录下的所有文件名，可以使用以下方法（本地数据这么写也不会有影响）
```python
file_list = self.remote.listdir(远程目录路径)
```


## frame_data说明
`frame_data`是一个`FrameData`类型对象，用于存储当前帧中的所有信息，`frame_data`的基本结构如下：

```python
frame_data = EasyDict({
    'pcd': {  # 点云数据，在此之下的所有数据都表示点云的属性，如颜色，标签，强度等。所有点云相关数据的行数必须与points的行数一致
        'points': None,  # 点云数据，是一个N x 3的numpy数组，表示点云的三维坐标
        'colors': None,  # 点云颜色数据，是一个N x 3的numpy数组，表示点云的RGB颜色值，如果为None，则使用默认渲染颜色
        ...  # 其他点云相关数据，注意，在pcd之下的数据的行数必须与points的行数一致，表示每个点的属性。例如intensity, label等
    },
    'geometry': {  # 几何元素数据，包含箭头，球体，三维包围盒。可视化时会绘制这些几何元素，空列表表示不绘制
        'arrows': [],  # 箭头
        'spheres': [],  # 球体
        'boxes': [],  # 三维包围盒
    },
    'pose': {  # 位姿信息，包含旋转矩阵和平移向量
        'R': None,  # 旋转矩阵
        'T': None,  # 平移向量
    }
    ... # 其他数据，可以根据需要随意添加
})

# 可以使用 frame_data.新增属性名 = 属性值 的方式添加其他数据，如
frame_data.frame_name = "frame_0"  # 添加帧名称
```