# 版权所有 (c) 2010-2022，InterDigital
# 保留所有权利。

# 请参阅根文件夹下的LICENSE。

import os  # 导入os库，用于操作系统相关的操作
import numpy as np  # 导入numpy库，numpy是Python的一个科学计算的库，提供了矩阵运算的功能
from torch.utils import data  # 导入torch.utils.data库，用于处理数据集
from pccai.utils.misc import pc_read  # 从pccai.utils.misc模块导入pc_read函数，用于读取点云数据

found_quantize = False  # 初始化found_quantize为False


def absoluteFilePaths(directory):  # 定义一个函数，用于获取目录下所有文件的绝对路径
   for dirpath, _, file_names in os.walk(directory):  # 遍历目录
       for f in file_names:  # 遍历文件名
           yield os.path.abspath(os.path.join(dirpath, f))  # 返回文件的绝对路径


class FordBase(data.Dataset):  # 定义一个名为FordBase的类，该类继承自data.Dataset类，用于处理Ford LiDAR数据集的基础类
    """A base Ford dataset."""

    def __init__(self, data_config, sele_config, **kwargs):  # 类的初始化方法

        base_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的目录

        # 数据集的通用选项
        self.return_intensity = data_config.get('return_intensity', False)  # 从数据配置中获取return_intensity参数，如果没有找到，则默认为False
        self.dataset_path = data_config.get('dataset_path', '../../datasets/ford/')  # 从数据配置中获取dataset_path参数，如果没有找到，则默认为'../../datasets/ford/'
        self.dataset_path = os.path.abspath(os.path.join(base_dir, self.dataset_path))  # 获取数据集的绝对路径
        self.translate = data_config.get('translate', [0, 0, 0])  # 从数据配置中获取translate参数，如果没有找到，则默认为[0, 0, 0]
        self.scale = data_config.get('scale', 1)  # 从数据配置中获取scale参数，如果没有找到，则默认为1
        self.point_max = data_config.get('point_max', -1)  # 从数据配置中获取point_max参数，如果没有找到，则默认为-1

        # 特定配置下的选项
        self.split = data_config[sele_config]['split']  # 从数据配置中获取split参数
        splitting = data_config['splitting'][self.split]  # 从数据配置中获取splitting参数

        self.im_idx = []  # 创建一个空的列表
        for i_folder in splitting:  # 遍历splitting
            folder_path = os.path.join(self.dataset_path, 'Ford_' + str(i_folder).zfill(2) + '_q_1mm')  # 获取文件夹路径
            assert os.path.exists(folder_path), f'{folder_path} does not exist'  # 如果文件夹不存在，则抛出异常
            self.im_idx += absoluteFilePaths(folder_path)  # 将文件夹中所有文件的绝对路径添加到列表中
        self.im_idx.sort()  # 对列表进行排序


    def __len__(self):  # 定义一个方法，用于返回样本的总数
        """Returns the total number of samples"""
        return len(self.im_idx)  # 返回列表的长度


    def __getitem__(self, index):  # 定义一个方法，用于获取指定索引的样本
        
        pc = (pc_read(self.im_idx[index]) + np.array(self.translate)) * self.scale  # 读取点云数据，并进行平移和缩放
        if self.point_max > 0 and pc.shape[0] > self.point_max:  # 如果point_max大于0且点云的数量大于point_max
                pc = pc[:self.point_max, :]  # 截取前point_max个点云
        return {'pc': pc, 'ref': None}  # 返回一个字典，包含点云和参考


    def get_pc_idx(self, index):  # 定义一个方法，用于获取指定索引的点云索引
        return self.im_idx[index]  # 返回指定索引的点云索引


class QnxadasBase(data.Dataset):  # 定义一个名为QnxadasBase的类，该类继承自data.Dataset类 是用于处理LiDAR数据集的基础类
    """A base Qnxadas dataset."""

    def __init__(self, data_config, sele_config, **kwargs):  # 类的初始化方法

        base_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的目录
        dataset_path_default = os.path.abspath(os.path.join(base_dir, '../../datasets/qnxadas/'))  # 获取默认的数据集路径

        # 数据集的通用选项
        self.return_intensity = data_config.get('return_intensity', False)  # 从数据配置中获取return_intensity参数，如果没有找到，则默认为False
        dataset_path = data_config.get('dataset_path', dataset_path_default)  # 从数据配置中获取dataset_path参数，如果没有找到，则默认为dataset_path_default
        self.translate = data_config.get('translate', [0, 0, 0])  # 从数据配置中获取translate参数，如果没有找到，则默认为[0, 0, 0]
        self.scale = data_config.get('scale', 1)  # 从数据配置中获取scale参数，如果没有找到，则默认为1

        # 特定配置下的选项
        self.split = data_config[sele_config]['split']  # 从数据配置中获取split参数
        splitting = data_config['splitting'][self.split]  # 从数据配置中获取splitting参数

        self.im_idx = []  # 创建一个空的列表
        for i_folder in splitting:  # 遍历splitting
            self.im_idx += absoluteFilePaths(os.path.join(dataset_path, i_folder))  # 将文件夹中所有文件的绝对路径添加到列表中
        self.im_idx.sort()  # 对列表进行排序


    def __len__(self):  # 定义一个方法，用于返回样本的总数
        """Returns the total number of samples"""
        return len(self.im_idx) // 2  # 返回列表的长度除以2


    def __getitem__(self, index):  # 定义一个方法，用于获取指定索引的样本
        pc = (pc_read(self.im_idx[2 * index + 1]) + np.array(self.translate)) * self.scale  # 读取点云数据，并进行平移和缩放
        return {'pc': pc, 'ref': None}  # 返回一个字典，包含点云和参考


    def get_pc_idx(self, index):  # 定义一个方法，用于获取指定索引的点云索引
        return self.im_idx[2 * index + 1]  # 返回指定索引的点云索引

 
class KITTIBase(data.Dataset):  # 定义一个名为KITTIBase的类，该类继承自data.Dataset类 是用于处理LiDAR数据集的基础类
    """A base SemanticKITTI dataset."""

    def __init__(self, data_config, sele_config, **kwargs):  # 类的初始化方法

        base_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的目录
        dataset_path = os.path.abspath(os.path.join(base_dir, '../../datasets/kitti/'))  # 获取数据集的路径

        # 其他特定选项
        self.translate = data_config.get('translate', [0, 0, 0])  # 从数据配置中获取translate参数，如果没有找到，则默认为[0, 0, 0]
        self.scale = data_config.get('scale', 1)  # 从数据配置中获取scale参数，如果没有找到，则默认为1
        self.quantize_resolution = data_config.get('quantize_resolution', None) if found_quantize else None  # 从数据配置中获取quantize_resolution参数，如果没有找到，则默认为None
        self.split = data_config[sele_config]['split']  # 从数据配置中获取split参数
        splitting = data_config['splitting'][self.split]  # 从数据配置中获取splitting参数
        
        self.im_idx = []  # 创建一个空的列表
        for i_folder in splitting:  # 遍历splitting
            self.im_idx += absoluteFilePaths('/'.join([dataset_path, str(i_folder).zfill(2),'velodyne']))  # 将文件夹中所有文件的绝对路径添加到列表中
        self.im_idx.sort()  # 对列表进行排序


    def __len__(self):  # 定义一个方法，用于返回样本的总数
        """Returns the total number of samples"""
        return len(self.im_idx)  # 返回列表的长度


    def __getitem__(self, index):  # 定义一个方法，用于获取指定索引的样本
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))  # 从文件中读取数据，并将其转换为numpy数组
        if self.quantize_resolution is not None:  # 如果quantize_resolution参数不为None
            pc = quantize_resolution(raw_data[:, :3], self.quantize_resolution)  # 对点云数据进行量化
        else:
            pc = (raw_data[:, :3] + np.array(self.translate)) * self.scale  # 对点云数据进行平移和缩放
        return {'pc': pc}  # 返回一个字典，包含点云


    def get_pc_idx(self, index):  # 定义一个方法，用于获取指定索引的点云索引
        return self.im_idx[index]  # 返回指定索引的点云索引

