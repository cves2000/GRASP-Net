# 导入所需的模块 这段代码定义了一个名为NNDFunction的类，该类继承自torch.autograd.Function，用于实现自定义的自动求导操作
import torch
from torch.autograd import Function
import my_lib_cuda as my_lib

# 定义一个名为NNDFunction的类，该类继承自torch.autograd.Function
class NNDFunction(Function):

    # 定义前向传播函数
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        # 获取输入数据的设备、批次大小和点的数量
        device = xyz1.device
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()   

        # 初始化距离和索引矩阵
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)
        
        # 根据输入数据是否在CUDA设备上，调用不同的函数计算最近邻距离和索引
        if not xyz1.is_cuda:
            my_lib.nnd_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            my_lib.nnd_forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        # 保存输入和输出数据，以便在反向传播时使用
        ctx.save_for_backward(xyz1,xyz2,dist1,dist2,idx1,idx2)
        idx1 = idx1.to(device=device, dtype=torch.long)
        idx2 = idx2.to(device=device, dtype=torch.long)
        return dist1, dist2, idx1, idx2

    # 定义反向传播函数
    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        # 获取保存的输入和输出数据
        xyz1, xyz2, dist1, dist2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        # 初始化梯度矩阵
        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())
        
        # 根据输入数据是否在CUDA设备上，调用不同的函数计算梯度
        if not graddist1.is_cuda:
            my_lib.nnd_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            my_lib.nnd_backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2
