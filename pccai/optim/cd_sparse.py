# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.  # 查看根文件夹下的许可证

# Compute Chamfer Distance loss for MinkowskiEngine sparse tensors  # 计算MinkowskiEngine稀疏张量的Chamfer距离损失

import torch  # 导入torch库
import sys  # 导入sys库
import os  # 导入os库

from pccai.optim.pcc_loss import PccLossBase  # 从pccai.optim.pcc_loss导入PccLossBase


def nndistance_simple(rec, data):  # 定义一个简单的最近邻搜索函数，不是很高效，仅供参考
    """
    A simple nearest neighbor search, not very efficient, just for reference
    """
    rec_sq = torch.sum(rec * rec, dim=2, keepdim=True)  # 计算rec的平方和
    data_sq = torch.sum(data * data, dim=2, keepdim=True)  # 计算data的平方和
    cross = torch.matmul(data, rec.permute(0, 2, 1))  # 计算data和rec的矩阵乘积
    dist = data_sq - 2 * cross + rec_sq.permute(0, 2, 1)  # 计算距离
    data_dist, data_idx = torch.min(dist, dim=2)  # 找到最小的距离和对应的索引
    rec_dist, rec_idx = torch.min(dist, dim=1)  # 找到最小的距离和对应的索引
    return data_dist, rec_dist, data_idx, rec_idx  # 返回结果


try:  # 尝试执行以下代码
    # If you want to use the efficient NN search for computing CD loss, compiled the nndistance()
    # function under the third_party folder according to instructions in Readme.md
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party/nndistance'))  # 添加路径到系统路径
    from modules.nnd import NNDModule  # 从modules.nnd导入NNDModule
    nndistance = NNDModule()  # 创建NNDModule实例
except ModuleNotFoundError:  # 如果上述代码执行出错（模块未找到）
    # Without the compiled nndistance(), by default the nearest neighbor will be done using pytorch-geometric
    nndistance = nndistance_simple  # 默认使用nndistance_simple函数进行最近邻搜索


class ChamferDistSparse(PccLossBase):  # 定义一个名为ChamferDistSparse的类，继承自PccLossBase
    """Chamfer distance loss for sparse voxels."""  # 用于稀疏体素的Chamfer距离损失

    def __init__(self, loss_args, syntax):  # 初始化函数
        super().__init__(loss_args, syntax)  # 调用父类的初始化函数

    def xyz_loss(self, loss_out, net_in, net_out):  # 定义xyz损失函数
        """Compute the xyz-loss."""  # 计算xyz损失

        x_hat = net_out['x_hat']  # 获取网络输出的x_hat
        gt = net_out['gt']  # 获取网络输出的gt
        batch_size = x_hat[-1][0].round().int().item() + 1  # 计算批次大小
        dist = torch.zeros(batch_size, device=x_hat.device)  # 初始化dist为全零张量
        for i in range(batch_size):  # 对每个批次进行循环
            dist_out, dist_x, _, _ = nndistance(  # 调用nndistance函数计算距离
                x_hat[x_hat[:, 0].round().int()==i, 1:].unsqueeze(0).contiguous(), 
                gt[gt[:, 0] == i, 1:].unsqueeze(0).float().contiguous()
            )
            dist[i] = torch.max(torch.mean(dist_out), torch.mean(dist_x))  # 计算最大距离
        loss = torch.mean(dist)  # 计算损失
        loss_out['xyz_loss'] = loss.unsqueeze(0)  # 将'xyz_loss'写入返回值

    def loss(self, net_in, net_out):  # 定义总损失函数
        """Overall R-D loss computation."""  # 总的R-D损失计算

        loss_out = {}  # 初始化损失输出

        # Rate loss
        if 'likelihoods' in net_out and len(net_out['likelihoods']) > 0:  # 如果网络输出中有'likelihoods'且其长度大于0
            self.bpp_loss(loss_out, net_out['likelihoods'], net_out['gt'].shape[0])  # 计算bpp损失
        else:  # 否则
            loss_out['bpp_loss'] = torch.zeros((1,))  # 将'bpp_loss'设置为0
            if net_out['x_hat'].is_cuda:  # 如果x_hat在cuda上
                loss_out['bpp_loss'] = loss_out['bpp_loss'].cuda()  # 将'bpp_loss'转移到cuda上
        
        # Distortion loss
        self.xyz_loss(loss_out, net_in, net_out)  # 计算xyz损失

        # R-D loss = alpha * D + beta * R
        loss_out["loss"] = self.alpha * loss_out['xyz_loss'] +  self.beta * loss_out["bpp_loss"]  # 计算R-D损失
        return loss_out  # 返回损失输出
        # 这段代码定义了一个名为ChamferDistSparse的类，该类继承自PccLossBase。这个类主要用于计算稀疏体素的Chamfer距离损失。
        # 其中，xyz_loss函数用于计算xyz损失，loss函数用于计算总的R-D损失。R-D损失是由alpha乘以xyz损失和beta乘以bpp损失得到的。
