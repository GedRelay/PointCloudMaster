# -*- coding: utf-8 -*-
"""
@Time        :  2024/6/5 13:02
@Author      :  GedRelay
@Email       :  gedrelay@stu.jnu.edu.cn
@Description :  filters, 一些常用的过滤函数，方便模块化实验
"""

class Filters():
    def __init__(self):
        pass

    @staticmethod
    def xyz2v(pcd_xyz, other_data, compensate=False):
        '''
        将三维空间点云转换为速度空间点云
        :param pcd_xyz: 点云 [N, 3]
        :param other_data: 其他数据，必须包含 rv 或者 vcps
        :param compensate: 是否是补偿速度
        :return: pcd_v, other_data
        '''
        import numpy as np

        if compensate:
            v = other_data['pointinfo-vcps']
        else:
            v = other_data['pointinfo-rv']

        pcd_xyz = pcd_xyz / np.linalg.norm(pcd_xyz, axis=1, keepdims=True)
        pcd_v = -pcd_xyz * v.reshape(-1, 1)

        return pcd_v, other_data

    @staticmethod
    def add_noise_v(pcd_xyz, other_data, mean=0, std=0.1, compensate=False):
        '''
        为速度添加高斯噪声
        :param pcd_xyz: 点云 [N, 3]
        :param other_data: 其他数据, 必须包含 rv 或者 vcps
        :param mean: 噪声均值
        :param std: 噪声标准差
        :param compensate: 是否是补偿速度
        :return: pcd_xyz, other_data
        '''
        import numpy as np

        if compensate:
            other_data['pointinfo-noise_v'] = np.random.normal(mean, std, other_data['pointinfo-vcps'].shape)
            other_data['pointinfo-vcps'] += other_data['pointinfo-noise_v']
        else:
            other_data['pointinfo-noise_v'] = np.random.normal(mean, std, other_data['pointinfo-rv'].shape)
            other_data['pointinfo-rv'] += other_data['pointinfo-noise_v']

        return pcd_xyz, other_data

    @staticmethod
    def add_noise_xyz(pcd_xyz, other_data, mean=0, std=0.2):
        '''
        在3d点云射线长度上添加高斯噪声
        :param pcd_xyz: 点云 [N, 3]
        :param other_data: 其他数据
        :param mean: 噪声均值
        :param std: 噪声标准差
        :return: pcd_xyz, other_data
        '''
        import numpy as np

        norm = np.linalg.norm(pcd_xyz, axis=1, keepdims=True)
        norm_noise = norm + np.random.normal(mean, std, norm.shape)
        pcd_xyz = pcd_xyz / norm * norm_noise

        return pcd_xyz, other_data

    @staticmethod
    def remove_points_by_mask(pcd_xyz, other_data, mask):
        '''
        通过mask去除点
        :param pcd_xyz: 点云
        :param other_data: 其他数据
        :param mask: 要去除的mask
        :return: pcd_xyz, other_data
        '''
        mask = ~mask

        for key in other_data.keys():
            if key.startswith('pointinfo-'):
                other_data[key] = other_data[key][mask]
        pcd_xyz = pcd_xyz[mask]

        return pcd_xyz, other_data

    @staticmethod
    def remain_points_by_mask(pcd_xyz, other_data, mask):
        '''
        通过mask保留点
        :param pcd_xyz: 点云
        :param other_data: 其他数据
        :param mask: 要保留的mask
        :return: pcd_xyz, other_data
        '''
        for key in other_data.keys():
            if key.startswith('pointinfo-'):
                other_data[key] = other_data[key][mask]
        pcd_xyz = pcd_xyz[mask]

        return pcd_xyz, other_data

    @staticmethod
    def remove_points_by_id(pcd_xyz, other_data, id_list):
        '''
        通过id去除点
        :param pcd_xyz: 点云
        :param other_data: 其他数据, 必须包含id
        :param id_list: 要去除的id列表
        :return: pcd_xyz, other_data
        '''
        import numpy as np

        ids = other_data['pointinfo-id']
        mask = np.zeros(ids.shape, dtype=bool)
        for id in id_list:
            mask = mask | (ids == id)

        pcd_xyz, other_data = Filters.remove_points_by_mask(pcd_xyz, other_data, mask)

        return pcd_xyz, other_data

    @staticmethod
    def remain_points_by_id(pcd_xyz, other_data, id_list):
        '''
        通过id保留点
        :param pcd_xyz: 点云
        :param other_data: 其他数据, 必须包含id
        :param id_list: 要保留的id列表
        :return: pcd_xyz, other_data
        '''
        import numpy as np

        ids = other_data['pointinfo-id']
        mask = np.zeros(ids.shape, dtype=bool)
        for id in id_list:
            mask = mask | (ids == id)

        pcd_xyz, other_data = Filters.remain_points_by_mask(pcd_xyz, other_data, mask)

        return pcd_xyz, other_data

    @staticmethod
    def remain_points_by_z_axis(pcd_xyz, other_data, z_min=-999, z_max=999):
        '''
        保留z轴在z_min和z_max之间的点
        :param pcd_xyz: 点云
        :param other_data: 其他数据
        :param z_min: z轴最小值
        :return: pcd_xyz, other_data
        '''

        mask = (pcd_xyz[:, 2] >= z_min) & (pcd_xyz[:, 2] <= z_max)

        pcd_xyz, other_data = Filters.remain_points_by_mask(pcd_xyz, other_data, mask)

        return pcd_xyz, other_data