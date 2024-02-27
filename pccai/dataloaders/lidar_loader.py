# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# A multi-modal data loader for LiDAR datasets.

import os
import numpy as np
from torch.utils import data

from pccai.utils.convert_image import pc2img
from pccai.utils.convert_octree import OctreeOrganizer
from pccai.dataloaders.lidar_base_loader import FordBase, KITTIBase, QnxadasBase


def get_base_lidar_dataset(data_config, sele_config):  # 定义一个函数，用于获取基础LiDAR数据集
    if data_config['dataset'].lower().find('ford') >= 0:  # 如果数据配置中的数据集名称包含'ford'
        loader_class = FordBase  # 使用FordBase类
    elif data_config['dataset'].lower().find('kitti') >= 0:  # 如果数据配置中的数据集名称包含'kitti'
        loader_class = KITTIBase  # 使用KITTIBase类
    elif data_config['dataset'].lower().find('qnxadas') >= 0:  # 如果数据配置中的数据集名称包含'qnxadas'
        loader_class = QnxadasBase  # 使用QnxadasBase类
    else:
        loader_class = None  # 如果数据配置中的数据集名称既不包含'ford'，也不包含'kitti'，也不包含'qnxadas'，则设置loader_class为None
    return loader_class(data_config, sele_config)  # 返回loader_class的实例


class LidarSimple(data.Dataset):  # 定义一个名为LidarSimple的类，该类继承自data.Dataset类 用于处理LiDAR数据集，它返回每个点云中指定数量的3D点
    """A simple LiDAR dataset which returns a specified number of 3D points in each point cloud."""

    def __init__(self, data_config, sele_config, **kwargs):  # 类的初始化方法

        self.point_cloud_dataset = get_base_lidar_dataset(data_config, sele_config)  # 获取基础LiDAR数据集
        self.num_points = data_config.get('num_points', 150000)  # 从数据配置中获取num_points参数，如果没有找到，则默认为150000
        self.seed = data_config.get('seed', None)  # 从数据配置中获取seed参数，如果没有找到，则默认为None
        self.sparse_collate = data_config.get('sparse_collate', False)  # 从数据配置中获取sparse_collate参数，如果没有找到，则默认为False
        self.voxelize = data_config.get('voxelize', False)  # 从数据配置中获取voxelize参数，如果没有找到，则默认为False

    def __len__(self):  # 定义一个方法，用于返回样本的总数
        return len(self.point_cloud_dataset)  # 返回点云数据集的长度
    
    def __getitem__(self, index):  # 定义一个方法，用于获取指定索引的样本
        pc = self.point_cloud_dataset[index]['pc']  # 获取点云数据集中指定索引的点云坐标
        np.random.seed(self.seed)  # 设置随机数种子
        if self.voxelize:  # 如果voxelize参数为True
            pc = np.round(pc[:self.num_points, :]).astype('int32')  # 对点云数据进行四舍五入，并转换为整型
            # 这是为了方便使用Minkowski Engine进行稀疏张量的构造
            if self.sparse_collate:  # 如果sparse_collate参数为True
                pc = np.hstack((np.zeros((pc.shape[0], 1), dtype='int32'), pc))  # 将点云数据和全为0的数组进行水平堆叠
                pc[0][0] = 1  # 将第一个元素设置为1
            return pc  # 返回点云数据
        else:
            choice = np.random.choice(pc.shape[0], self.num_points, replace=True)  # 从点云数据中随机选择num_points个点
            return pc[choice, :].astype(dtype=np.float32)  # 返回选择的点云数据，并将其转换为浮点型

class LidarSpherical(data.Dataset):  # 定义一个名为LidarSpherical的类，该类继承自data.Dataset类
    """Converts the original Cartesian coordinate to spherical coordinate then represent as 2D images."""

    def __init__(self, data_config, sele_config, **kwargs):  # 类的初始化方法

        self.point_cloud_dataset = get_base_lidar_dataset(data_config, sele_config)  # 获取基础LiDAR数据集
        self.width = data_config['spherical_cfg'].get('width', 1024)  # 从数据配置中获取width参数，如果没有找到，则默认为1024
        self.height = data_config['spherical_cfg'].get('height', 128)  # 从数据配置中获取height参数，如果没有找到，则默认为128
        self.v_fov = data_config['spherical_cfg'].get('v_fov', [-28, 3.0])  # 从数据配置中获取v_fov参数，如果没有找到，则默认为[-28, 3.0]
        self.h_fov = data_config['spherical_cfg'].get('h_fov', [-180, 180])  # 从数据配置中获取h_fov参数，如果没有找到，则默认为[-180, 180]
        self.origin_shift = data_config['spherical_cfg'].get('origin_shift', [0, 0, 0])  # 从数据配置中获取origin_shift参数，如果没有找到，则默认为[0, 0, 0]
        self.v_fov, self.h_fov = np.array(self.v_fov) / 180 * np.pi, np.array(self.h_fov) / 180 * np.pi  # 将v_fov和h_fov参数转换为弧度制
        self.num_points = self.width * self.height  # 计算点的数量
        self.inf = 1e6  # 设置inf参数为1e6

    def __len__(self):  # 定义一个方法，用于返回样本的总数
        return len(self.point_cloud_dataset)  # 返回点云数据集的长度

    def __getitem__(self, index):  # 定义一个方法，用于获取指定索引的样本
        data = self.point_cloud_dataset[index]['pc']  # 获取点云数据集中指定索引的点云坐标
        data[:, 0] += self.origin_shift[0]  # 对点云数据的x坐标进行平移
        data[:, 1] += self.origin_shift[1]  # 对点云数据的y坐标进行平移
        data[:, 2] += self.origin_shift[2]  # 对点云数据的z坐标进行平移
        data_img = pc2img(self.h_fov, self.v_fov, self.width, self.height, self.inf, data)  # 将点云数据转换为图像        

        return data_img  # 返回图像


class LidarOctree(data.Dataset):  # 定义一个名为LidarOctree的类，该类继承自data.Dataset类
    """Converts an original point cloud into an octree."""

    def __init__(self, data_config, sele_config, **kwargs):  # 类的初始化方法

        self.point_cloud_dataset = get_base_lidar_dataset(data_config, sele_config)  # 获取基础LiDAR数据集
        self.rw_octree = data_config.get('rw_octree', False)  # 从数据配置中获取rw_octree参数，如果没有找到，则默认为False
        if self.rw_octree:  # 如果rw_octree参数为True
            self.rw_partition_scheme = data_config.get('rw_partition_scheme', 'default')  # 从数据配置中获取rw_partition_scheme参数，如果没有找到，则默认为'default'
        self.octree_cache_folder = 'octree_cache'  # 设置octree_cache_folder参数为'octree_cache'

        # 创建一个八叉树格式器，用于将八叉树组织成数组
        self.octree_organizer = OctreeOrganizer(
            data_config['octree_cfg'],  # 从数据配置中获取octree_cfg参数
            data_config[sele_config].get('max_num_points', 150000),  # 从数据配置中获取max_num_points参数，如果没有找到，则默认为150000
            kwargs['syntax'].syntax_gt,  # 从关键字参数中获取syntax_gt参数
            self.rw_octree,  # 设置rw_octree参数
            data_config[sele_config].get('shuffle_blocks', False),  # 从数据配置中获取shuffle_blocks参数，如果没有找到，则默认为False
        )

    def __len__(self):  # 定义一个方法，用于返回样本的总数
        return len(self.point_cloud_dataset)  # 返回点云数据集的长度

    def __getitem__(self, index):  # 定义一个方法，用于获取指定索引的样本
    
        if self.rw_octree:  # 如果rw_octree参数为True
            file_name = os.path.relpath(self.point_cloud_dataset.get_pc_idx(index), self.point_cloud_dataset.dataset_path)  # 获取点云数据集中指定索引的点云索引相对于数据集路径的路径
            file_name = os.path.join(self.point_cloud_dataset.dataset_path, self.octree_cache_folder, self.rw_partition_scheme, file_name)  # 获取文件名
            file_name = os.path.splitext(file_name)[0] + '.pkl'  # 将文件名的扩展名更改为'.pkl'
        else: 
            file_name = None  # 如果rw_octree参数为False，将file_name设置为None
    
        pc = self.point_cloud_dataset[index]['pc']  # 获取点云数据集中指定索引的点云坐标
        # 执行八叉树划分并组织数据
        pc_formatted, _, _, _, _ = self.octree_organizer.organize_data(pc, file_name=file_name)  # 使用八叉树组织器对点云数据进行组织
    
        return pc_formatted  # 返回组织后的点云数据

