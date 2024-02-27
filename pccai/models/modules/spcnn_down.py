# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Downsample with sparse CNN 稀疏卷积神经网络的下采样

# 导入所需的库
import os
import sys
import torch
import torch.nn as nn
import MinkowskiEngine as ME

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../third_party/PCGCv2'))
from autoencoder import InceptionResNet, make_layer

# 定义一个函数，用于创建基于IRN的下采样块
def make_sparse_down_block(in_dim, hidden_dim, out_dim, doLastRelu=False):
    """
    创建一个基于IRN的下采样块
    """
    layers = [
        # 创建一个Minkowski卷积层
        ME.MinkowskiConvolution(in_channels=in_dim, out_channels=hidden_dim,
            kernel_size=3, stride=1, bias=True, dimension=3),
        # 创建一个MinkowskiReLU层
        ME.MinkowskiReLU(inplace=True),
        # 创建一个Minkowski卷积层
        ME.MinkowskiConvolution(in_channels=hidden_dim, out_channels=out_dim,
            kernel_size=2, stride=2, bias=True, dimension=3),
        # 创建一个MinkowskiReLU层
        ME.MinkowskiReLU(inplace=True),
        # 创建一个InceptionResNet层
        make_layer(block=InceptionResNet, block_layers=3, channels=out_dim),
    ]
    # 如果doLastRelu为True，则在最后添加一个MinkowskiReLU层
    if doLastRelu: layers.append(ME.MinkowskiReLU(inplace=True))
    # 返回一个顺序容器，其中包含了上述所有的层
    return torch.nn.Sequential(*layers)

# 定义SparseCnnDown1类，它是一个进行一次下采样的稀疏CNN模块
class SparseCnnDown1(nn.Module):
    """
    SparseCnnDown模块：进行一次下采样
    """

    def __init__(self, net_config, **kwargs):
        super(SparseCnnDown1, self).__init__()

        # 从net_config中获取dims的值
        self.dims = net_config['dims']
        # 创建一个下采样块和一个Minkowski卷积层
        self.down_block0 = make_sparse_down_block(self.dims[0], self.dims[0], self.dims[1], True)
        self.conv_last = ME.MinkowskiConvolution(in_channels=self.dims[1], out_channels=self.dims[2],
            kernel_size=3, stride=1, bias=True, dimension=3)

    def forward(self, x):

        # 将输入数据x传递给下采样块进行处理
        out0 = self.down_block0(x)
        # 将处理后的数据传递给Minkowski卷积层进行处理
        out0 = self.conv_last(out0)
        return out0

# 定义SparseCnnDown2类，它是一个进行两次下采样的稀疏CNN模块
class SparseCnnDown2(nn.Module):
    """
    SparseCnnDown模块：进行两次下采样
    """

    def __init__(self, net_config, **kwargs):
        super(SparseCnnDown2, self).__init__()

        # 从net_config中获取dims的值
        self.dims = net_config['dims']
        # 创建两个下采样块
        self.down_block2 = make_sparse_down_block(self.dims[0], self.dims[0], self.dims[1], True)
        self.down_block1 = make_sparse_down_block(self.dims[1], self.dims[1], self.dims[2], False)

    def forward(self, x):

        # 第一次下采样
        out2 = self.down_block2(x)
        # 第二次下采样
        out1 = self.down_block1(out2)
        return out1
