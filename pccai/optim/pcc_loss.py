# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder. 这段代码定义了一个名为PccLossBase的类，该类是用于计算点云压缩的速率失真损失的基类。
#其中，bpp_loss函数用于计算速率损失，xyz_loss函数和loss函数需要在子类中实现


import math  # 导入数学库
import torch  # 导入PyTorch库
import sys  # 导入系统库
import os  # 导入操作系统库

class PccLossBase:  # 定义一个名为PccLossBase的类
    """A base class of rate-distortion loss computation for point cloud compression."""
    # 用于计算点云压缩的速率失真损失的基类

    def __init__(self, loss_args, syntax):  # 初始化函数
        self.alpha = loss_args['alpha']  # 设置alpha参数
        self.beta = loss_args['beta']  # 设置beta参数
        self.hetero = syntax.hetero  # 设置hetero参数
        self.phase = syntax.phase  # 设置phase参数

    @staticmethod
    def bpp_loss(loss_out, likelihoods, count):  # 定义bpp损失函数
        """Compute the rate loss with the likelihoods."""
        # 使用似然性计算速率损失

        bpp_loss = 0  # 初始化bpp损失为0
        for k, v in likelihoods.items():  # 对似然性字典进行循环
            if v is not None:  # 如果v不为空
                loss = torch.log(v).sum() / (-math.log(2) * count)  # 计算损失
                bpp_loss += loss  # 累加损失
                loss_out[f'bpp_loss_{k}'] = loss.unsqueeze(0)  # 将损失写入返回值
        loss_out['bpp_loss'] = bpp_loss.unsqueeze(0)  # 将bpp损失写入返回值

    def xyz_loss(self, **kwargs):  # 定义xyz损失函数
        """Needs to implement the xyz_loss"""
        # 需要实现xyz损失

        raise NotImplementedError()  # 抛出未实现错误

    def loss(self, **kwargs):  # 定义总损失函数
        """Needs to implement the overall loss. Can be R-D loss for lossy compression, or rate-only loss for lossless compression."""
        # 需要实现总损失。可以是有损压缩的R-D损失，也可以是无损压缩的仅速率损失。

        raise NotImplementedError()  # 抛出未实现错误
