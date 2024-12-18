# -*- coding: utf-8 -*-
"""
@Time        :  2024/6/4 22:15
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  tools 工具
"""

class Tools():
    def __init__(self):
        pass

    @staticmethod
    def get_bbox_from_points(points, color=(1, 0, 0), oriented=True):
        '''
        从点云中获取包围盒
        :param points: 点云 (N, 3)
        :param color: 包围盒颜色
        :param oriented: 是否为旋转包围盒
        :return: 包围盒 o3d.geometry.OrientedBoundingBox
        '''
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if oriented:
            bbox = pcd.get_oriented_bounding_box()
        else:
            bbox = pcd.get_axis_aligned_bounding_box()
        bbox.color = color
        return bbox

    @staticmethod
    def get_bbox_by_corners(corners, color=(1, 0, 0)):
        '''
        从8个点获取包围盒
        :param corners: 8个点 (8, 3)
        :param color: 包围盒颜色
        :return: 包围盒 o3d.geometry.OrientedBoundingBox
        '''
        import open3d as o3d

        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom square
            [4, 5], [5, 6], [6, 7], [7, 4],  # top square
            [0, 4], [1, 5], [2, 6], [3, 7]  # vertical lines
        ]

        bbox = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(lines),
        )
        bbox.colors = o3d.utility.Vector3dVector([color] * len(lines))
        bbox.paint_uniform_color(color)
        return bbox

    @staticmethod
    def get_arrow(vector, start, color=(1, 0, 0)):
        '''
        获取箭头
        :param vector: 矢量
        :param start: 起始点
        :param color: 颜色
        :return: 箭头 o3d.geometry.TriangleMesh
        '''
        import open3d as o3d
        import numpy as np

        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.04,
                                                       cone_radius=0.06,
                                                       cylinder_height=0.8,
                                                       cone_height=0.2)
        arrow.paint_uniform_color(color)

        # 计算z轴单位向量(0, 0, 1)到vector的旋转矩阵
        vector_len = np.linalg.norm(vector)
        v = vector / vector_len
        z = np.array([0, 0, 1])
        k = np.cross(z, v)  # 转轴 k = z x v

        if np.linalg.norm(k) == 0:
            rotation_matrix = np.eye(3)
        else:
            k = k / np.linalg.norm(k)

            # 计算旋转角度 θ
            cos_theta = np.dot(z, v)
            theta = np.arccos(cos_theta)

            # 构造旋转矩阵
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])

            rotation_matrix = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

        arrow.rotate(rotation_matrix * vector_len, (0, 0, 0))  # 旋转
        arrow.translate(start)  # 平移
        return arrow

    @staticmethod
    def euler2mat(roll, pitch, yaw, degrees=True):
        '''
        欧拉角转旋转矩阵
        :param roll: 滚转角
        :param pitch: 俯仰角
        :param yaw: 偏航角
        :param degrees: 是否为角度制
        :return:
        '''
        import numpy as np
        import math

        # 角度制转弧度制
        if degrees:
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)

        cos_r, sin_r = math.cos(roll), math.sin(roll)
        R_x = np.array([[1, 0, 0],
                        [0, cos_r, -sin_r],
                        [0, sin_r, cos_r]])

        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        R_y = np.array([[cos_p, 0, sin_p],
                        [0, 1, 0],
                        [-sin_p, 0, cos_p]])

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        R_z = np.array([[cos_y, -sin_y, 0],
                        [sin_y, cos_y, 0],
                        [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    @staticmethod
    def get_id_times(ids):
        '''
        获取每个id的出现次数
        :param ids: numpy.array (N,)
        :return: id_times numpy.array (N, 2), 第一列为id, 第二列为出现次数, 按照出现次数降序排列
        '''
        import numpy as np

        id_times = np.unique(ids, return_counts=True)
        id_times = np.vstack(id_times).T
        id_times = id_times[id_times[:, 1].argsort()[::-1]]
        id_times = id_times.astype(np.int32)

        return id_times

    @staticmethod
    def get_sphere(center, radius):
        '''
        获取球体
        :param center: 球心
        :param radius: 半径
        :return: 球体 o3d.geometry.TriangleMesh
        '''
        import open3d as o3d

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(center)
        return sphere

    @staticmethod
    def dbscan(X, eps=0.5, min_samples=5):
        '''
        dbscan聚类
        :param X: 数据 [N, D]
        :param eps: 邻域距离
        :param min_samples: 最小样本数
        :return: labels numpy.array (N,), -1表示离群点
        '''
        if X.shape[0] == 0:
            return []

        from sklearn.cluster import DBSCAN

        estimator = DBSCAN(eps=eps,
                           min_samples=min_samples,
                            metric='euclidean',
                            algorithm='auto',
                            leaf_size=64,
                            n_jobs=12)

        estimator.fit(X)
        labels = estimator.labels_

        return labels

    @staticmethod
    def dbscan2(X, eps=0.5, min_samples=5):
        '''
        dbscan聚类
        :param X: 数据 [N, D]
        :param eps: 邻域距离
        :param min_samples: 最小样本数
        :return: labels numpy.array (N,), -1表示离群点
        '''
        if X.shape[0] == 0:
            return []

        import numpy as np
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X)

        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False))

        return labels

    @staticmethod
    def mean_shift(X, bandwidth=0.5):
        '''
        mean shift聚类
        :param X: 数据 [N, D]
        :param bandwidth: 带宽
        :return: labels numpy.array (N,)
        '''
        if X.shape[0] == 0:
            return []

        from sklearn.cluster import MeanShift

        estimator = MeanShift(bandwidth=bandwidth,
                              bin_seeding=True,
                              min_bin_freq=1,
                              cluster_all=True,
                              n_jobs=4)

        estimator.fit(X)
        labels = estimator.labels_

        return labels

    @staticmethod
    def kmeans(X, n_clusters=8):
        '''
        kmeans聚类
        :param X: 数据 [N, D]
        :param n_clusters: 聚类数
        :return: labels numpy.array (N,)
        '''
        if X.shape[0] == 0:
            return []

        from sklearn.cluster import KMeans

        estimator = KMeans(n_clusters=n_clusters,
                           init='k-means++',
                           n_init=10,
                           max_iter=300,
                           tol=1e-4,
                           verbose=0,
                           random_state=None,
                           copy_x=True,
                           algorithm='lloyd')

        estimator.fit(X)
        labels = estimator.labels_

        return labels

    @staticmethod
    def xyz2abrho(pcd_xyz):
        '''
        将三维空间点云转换为极坐标空间点云
        :param pcd_xyz: 点云 [N, 3]
        :return: pcd_abrho
        '''
        import numpy as np

        a = np.arctan2(pcd_xyz[:, 1], pcd_xyz[:, 0])
        b = np.arctan2(pcd_xyz[:, 2], np.linalg.norm(pcd_xyz[:, :2], axis=1))
        rho = np.linalg.norm(pcd_xyz, axis=1)

        pcd_abrho = np.vstack([a, b, rho]).T

        return pcd_abrho

    @staticmethod
    def inverse_rigid_trans(Tr):
        '''
        逆刚体变换矩阵
        :param Tr: 刚体变换矩阵 (3x4 as [R|t])
        :return: inv_Tr: 逆刚体变换矩阵 (3x4)
        '''
        import numpy as np
        inv_Tr = np.zeros_like(Tr) # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr


























