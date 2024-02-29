# Copyright (c) 2010-2022, InterDigital  # 版权声明，代码所有权归InterDigital公司所有
# All rights reserved. 

# See LICENSE under the root folder.  # 指向许可证文件的注释

# MLP Decoder implemented with MinkowskiEngine operated on sparse tensors  # 使用MinkowskiEngine实现的多层感知器（MLP）解码器，主要用于处理稀疏张量。

import torch  # 导入PyTorch库，用于创建和操作张量
import torch.nn as nn  # 导入PyTorch的神经网络库
import numpy as np  # 导入NumPy库，用于数值计算
import MinkowskiEngine as ME  # 导入MinkowskiEngine库，用于处理稀疏张量

def make_pointwise_mlp_sparse(dims, doLastRelu=False):  # 定义一个函数，用于创建基于MinkowskiEngine的逐点MLP层
    layers = []  # 创建一个空的层列表
    for i in range(len(dims) - 1):  # 对每个维度进行循环
        layers.append(
            ME.MinkowskiLinear(dims[i], dims[i + 1], bias=True)  # 在每个维度之间添加一个MinkowskiLinear层
        )
        if i != len(dims) - 2 or doLastRelu:  # 如果不是最后一层或者doLastRelu为真
            layers.append(ME.MinkowskiReLU(inplace=True))  # 添加一个MinkowskiReLU层
    return torch.nn.Sequential(*layers)  # 使用torch.nn.Sequential将所有层打包成一个序列

class MlpDecoderSparse(nn.Module):  # 定义一个名为MlpDecoderSparse的类，继承自torch.nn.Module
    def __init__(self, net_config, **kwargs):  # 类的初始化函数
        super(MlpDecoderSparse, self).__init__()  # 调用父类的初始化函数
        self.num_points = net_config['num_points']  # 从网络配置中获取点的数量
        dims = net_config['dims']  # 从网络配置中获取维度
        self.mlp = make_pointwise_mlp_sparse(dims + [3 * self.num_points], doLastRelu=False)  # 使用make_pointwise_mlp_sparse函数创建MLP层

    def forward(self, x):  # 类的前向传播函数
        out = self.mlp(x)  # 将输入x传递给MLP层
        return out  # 返回输出
