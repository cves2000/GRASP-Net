# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


import torch.nn as nn
from pccai.optim.utils import get_loss_class

# Import all the architectures to be used
from pccai.models.architectures.grasp import GeoResCompression

# List the all the architectures in the following dictionary 
# For a custom architecture, it is recommended to implement a compress() and a decompress() functions that can be called by the codec.
architectures = {
    'grasp': GeoResCompression,
}
定义了一个字典 architectures，其中包含了所有可用的神经网络架构。在这个例子中，只有一个架构 'grasp': GeoResCompression。

def get_architecture_class(architecture_name):
    architecture = architectures.get(architecture_name.lower(), None)
    assert architecture is not None, f'architecture "{architecture_name}" not found, valid architectures are: {list(architectures.keys())}'
    return architecture
这个函数接收一个架构名称作为输入，然后在 architectures 字典中查找对应的架构类。如果找不到对应的架构，就会抛出一个断言错误。

class PccModelWithLoss(nn.Module):  # 定义一个名为PccModelWithLoss的类，该类继承自nn.Module类
    """A wrapper class for point cloud compression model and its associated loss function."""

    def __init__(self, net_config, syntax, loss_args = None):  # 类的初始化方法

        super(PccModelWithLoss, self).__init__()  # 调用父类的初始化方法

        # 获取架构并初始化它
        architecture_class = get_architecture_class(net_config['architecture'])  # 获取架构类
        self.pcc_model = architecture_class(net_config['modules'], syntax)  # 初始化点云压缩模型

        # 获取损失类并初始化它
        if loss_args is not None:  # 如果损失参数不为None
            loss_class = get_loss_class(loss_args['loss'])  # 获取损失类
            self.loss = loss_class(loss_args, syntax)  # 初始化损失函数
    
    def forward(self, data):  # 定义一个方法，用于前向传播
        out = self.pcc_model(data)  # 对数据进行点云压缩
        if self.loss is not None: out['loss'] = self.loss.loss(data, out)  # 如果损失函数不为None，计算损失

        return out  # 返回输出
