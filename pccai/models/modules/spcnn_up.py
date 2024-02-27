# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Upsample with sparse CNN 实现了稀疏卷积神经网络的上采样

# 导入所需的库
import os
import sys
import torch
import torch.nn as nn
import MinkowskiEngine as ME

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../third_party/PCGCv2'))
from data_utils import isin
from autoencoder import InceptionResNet, make_layer

# 定义一个函数，用于创建基于IRN的上采样块
def make_sparse_up_block(in_dim, hidden_dim, out_dim, doLastRelu):
    """
    创建一个基于IRN的上采样块
    """
    layers = [
        # 创建一个Minkowski生成卷积转置层
        ME.MinkowskiGenerativeConvolutionTranspose(in_channels=in_dim, out_channels=hidden_dim,
            kernel_size=2, stride=2, bias=True, dimension=3),
        # 创建一个MinkowskiReLU层
        ME.MinkowskiReLU(inplace=True),
        # 创建一个Minkowski卷积层
        ME.MinkowskiConvolution(in_channels=hidden_dim, out_channels=out_dim,
            kernel_size=3, stride=1, bias=True, dimension=3),
        # 创建一个MinkowskiReLU层
        ME.MinkowskiReLU(inplace=True),
        # 创建一个InceptionResNet层
        make_layer(block=InceptionResNet, block_layers=3, channels=out_dim),
    ]
    # 如果doLastRelu为True，则在最后添加一个MinkowskiReLU层
    if doLastRelu: layers.append(ME.MinkowskiReLU(inplace=True))
    # 返回一个顺序容器，其中包含了上述所有的层
    return torch.nn.Sequential(*layers)

# 定义SparseCnnUp1类，它是一个进行一次上采样的稀疏CNN模块
class SparseCnnUp1(nn.Module):
    """
    SparseCnnUp模块：进行一次上采样
    """

    def __init__(self, net_config, **kwargs):
        super(SparseCnnUp1, self).__init__()

        # 从net_config中获取dims的值
        self.dims = net_config['dims']
        # 创建一个上采样块
        self.up_block0 = make_sparse_up_block(self.dims[0], self.dims[1], self.dims[1], False)
        # 创建一个Minkowski剪枝层
        self.pruning = ME.MinkowskiPruning()

    def forward(self, y1, gt_pc): # 从粗到细

        # 将输入数据y1传递给上采样块进行处理
        out = self.up_block0(y1)
        # 对处理后的数据进行剪枝
        out = self.prune_voxel(out, gt_pc.C)
        return out

    def prune_voxel(self, coarse_voxels, refined_voxels):
        # 创建一个掩码，用于标记在refined_voxels中存在的coarse_voxels
        mask = isin(coarse_voxels.C, refined_voxels)
        # 对coarse_voxels进行剪枝
        data_pruned = self.pruning(coarse_voxels, mask.to(coarse_voxels.device))
        return data_pruned

# 定义SparseCnnUp2类，它是一个进行两次上采样的稀疏CNN模块
class SparseCnnUp2(nn.Module):
    """
    SparseCnnUp2模块：进行两次上采样
    """

    def __init__(self, net_config, **kwargs):
        super(SparseCnnUp2, self).__init__()

        # 从net_config中获取dims的值
        self.dims = net_config['dims']
        # 创建两个上采样块
        self.up_block1 = make_sparse_up_block(self.dims[0], self.dims[1], self.dims[1], True)
        self.up_block2 = make_sparse_up_block(self.dims[1], self.dims[2], self.dims[2], False)

        # 创建一个Minkowski剪枝层和一个Minkowski最大池化层
        self.pruning = ME.MinkowskiPruning()
        self.pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)

    def forward(self, y1, gt_pc): # 从粗到细

        # 第一次上采样
        out = self.up_block1(y1)
        y2_C = self.pool(gt_pc)
        out = SparseCnnUp1.prune_voxel(self, out, y2_C.C)

        # 第二次上采样
        out = self.up_block2(out)
        out = SparseCnnUp1.prune_voxel(self, out, gt_pc.C)

        return out
