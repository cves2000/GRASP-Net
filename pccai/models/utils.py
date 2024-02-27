# 导入所需的库
import numpy as np
import torch
import torch.nn as nn

# 定义一个函数，用于获取一系列的2D卷积层
def get_Conv2d_layer(dims, kernel_size, stride, doLastRelu):
    """Elementary 2D convolution layers."""
    layers = []
    for i in range(1, len(dims)):
        # 计算填充大小，如果卷积核大小不为1，则填充大小为(kernel_size - 1) / 2
        padding = int((kernel_size - 1) / 2) if kernel_size != 1 else 0
        # 添加卷积层
        layers.append(nn.Conv2d(in_channels=dims[i-1], out_channels=dims[i],
            kernel_size=kernel_size, stride=stride, padding=padding, bias=True))
        # 如果是最后一层且不需要ReLU激活函数，则跳过
        if i==len(dims)-1 and not doLastRelu:
            continue
        # 添加ReLU激活函数
        layers.append(nn.ReLU(inplace=True))
    return layers

# 定义一个类，用于创建一系列的2D卷积层
class Conv2dLayers(nn.Sequential):
    """2D convolutional layers."""
    def __init__(self, dims, kernel_size, doLastRelu=False):
        layers = get_Conv2d_layer(dims, kernel_size, 1, doLastRelu)
        super(Conv2dLayers, self).__init__(*layers)

# 定义一个函数，用于获取并初始化全连接层
def get_and_init_FC_layer(din, dout, init_bias='zeros'):
    """Get a fully-connected layer."""
    li = nn.Linear(din, dout)
    # 初始化权重/偏置
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
    if init_bias == 'uniform':
        nn.init.uniform_(li.bias)
    elif init_bias == 'zeros':
        li.bias.data.fill_(0.)
    else:
        raise 'Unknown init ' + init_bias
    return li

# 定义一个函数，用于获取一系列的多层感知机（MLP）层
def get_MLP_layers(dims, doLastRelu, init_bias='zeros'):
    """Get a series of MLP layers."""
    layers = []
    for i in range(1, len(dims)):
        # 添加并初始化全连接层
        layers.append(get_and_init_FC_layer(dims[i-1], dims[i], init_bias=init_bias))
        # 如果是最后一层且不需要ReLU激活函数，则跳过
        if i==len(dims)-1 and not doLastRelu:
            continue
        # 添加ReLU激活函数
        layers.append(nn.ReLU())
    return layers



# 定义一个类，用于创建一系列的点对点的多层感知机（MLP）层
class PointwiseMLP(nn.Sequential):
    def __init__(self, dims, doLastRelu=False, init_bias='zeros'):
        layers = get_MLP_layers(dims, doLastRelu, init_bias)
        super(PointwiseMLP, self).__init__(*layers)

# 定义一个全局池化类
class GlobalPool(nn.Module):
    def __init__(self, pool_layer):
        super(GlobalPool, self).__init__()
        self.Pool = pool_layer

    def forward(self, X):
        X = X.unsqueeze(-3) #Bx1xNxK
        X = self.Pool(X)
        X = X.squeeze(-2)
        X = X.squeeze(-2)   #BxK
        return X

# 定义一个类，用于创建PointNet的全局最大池化层
class PointNetGlobalMax(nn.Sequential):
    def __init__(self, dims, doLastRelu=False):
        layers = [
            PointwiseMLP(dims, doLastRelu=doLastRelu),      #BxNxK
            GlobalPool(nn.AdaptiveMaxPool2d((1, dims[-1]))),#BxK
        ]
        super(PointNetGlobalMax, self).__init__(*layers)

# 定义一个类，用于创建PointNet的全局平均池化层
class PointNetGlobalAvg(nn.Sequential):
    def __init__(self, dims, doLastRelu=True):
        layers = [
            PointwiseMLP(dims, doLastRelu=doLastRelu),      #BxNxK
            GlobalPool(nn.AdaptiveAvgPool2d((1, dims[-1]))),#BxK
        ]
        super(PointNetGlobalAvg, self).__init__(*layers)

# 定义一个类，用于创建原始的PointNet模型
class PointNet(nn.Sequential):
    def __init__(self, MLP_dims, FC_dims, MLP_doLastRelu):
        assert(MLP_dims[-1]==FC_dims[0])
        layers = [
            PointNetGlobalMax(MLP_dims, doLastRelu=MLP_doLastRelu),#BxK
        ]
        layers.extend(get_MLP_layers(FC_dims, False))
        super(PointNet, self).__init__(*layers)
