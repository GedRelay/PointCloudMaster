# -*- coding: utf-8 -*-
import numpy as np

class Filters():
    @staticmethod
    def remove_points_by_mask(frame_data, mask):
        '''
        通过mask去除点
        :param frame_data: 帧数据
        :param mask: 要去除的mask
        :return: frame_data
        '''
        mask = ~mask
        for key in frame_data.pcd.keys():
            data = frame_data.pcd[key]
            if data is not None:
                frame_data.pcd[key] = data[mask]
        return frame_data


    @staticmethod
    def remain_points_by_mask(frame_data, mask):
        '''
        通过mask保留点
        :param pcd_xyz: 点云
        :param mask: 要保留的mask
        :return: frame_data
        '''
        for key in frame_data.pcd.keys():
            data = frame_data.pcd[key]
            if data is not None:
                frame_data.pcd[key] = data[mask]
        return frame_data
    

    @staticmethod
    def remove_points_by_id(frame_data, id_list):
        '''
        通过id去除点
        :param frame_data: 帧数据
        :param id_list: 要去除的id列表
        :return: frame_data
        '''
        if isinstance(id_list, int):
            id_list = [id_list]
        ids = frame_data.pcd.id
        mask = np.zeros(ids.shape, dtype=bool)
        for id in id_list:
            mask = mask | (ids == id)
        frame_data = Filters.remove_points_by_mask(frame_data, mask)
        return frame_data


    @staticmethod
    def remain_points_by_id(frame_data, id_list):
        '''
        通过id保留点
        :param frame_data: 帧数据
        :param id_list: 要保留的id列表
        :return: frame_data
        '''
        if isinstance(id_list, int):
            id_list = [id_list]
        ids = frame_data.pcd.id
        mask = np.zeros(ids.shape, dtype=bool)
        for id in id_list:
            mask = mask | (ids == id)
        frame_data = Filters.remain_points_by_mask(frame_data, mask)
        return frame_data


    @staticmethod
    def remain_points_by_z_axis(frame_data, z_min=-999, z_max=999):
        '''
        保留z轴在z_min和z_max之间的点
        :param frame_data: 帧数据
        :param z_min: z轴最小值
        :param z_max: z轴最大值
        :return: frame_data
        '''
        mask = (frame_data.pcd.points[:, 2] >= z_min) & (frame_data.pcd.points[:, 2] <= z_max)
        frame_data = Filters.remain_points_by_mask(frame_data, mask)
        return frame_data


    @staticmethod
    def remain_points_by_range(frame_data, range):
        '''
        保留范围内的点
        :param frame_data: 帧数据
        :param range: [x_min, y_min, z_min, x_max, y_max, z_max]
        :return: frame_data
        '''
        assert len(range) == 6
        mask = (frame_data.pcd.points[:, 0] >= range[0]) & (frame_data.pcd.points[:, 0] <= range[3]) \
               & (frame_data.pcd.points[:, 1] >= range[1]) & (frame_data.pcd.points[:, 1] <= range[4]) \
               & (frame_data.pcd.points[:, 2] >= range[2]) & (frame_data.pcd.points[:, 2] <= range[5])

        frame_data = Filters.remain_points_by_mask(frame_data, mask)

        return frame_data