# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Utility functions for sparse tensors

# 导入所需的库
import torch
import numpy as np
import MinkowskiEngine as ME

# 定义一个函数，用于对稀疏张量进行缩放
def scale_sparse_tensor_batch(x, factor):
    # 将x的坐标乘以缩放因子并四舍五入为整数
    coords = torch.hstack((x.C[:,0:1], (x.C[:,1:]*factor).round().int()))
    # 创建一个新的稀疏张量，其中的特征是全1，坐标是缩放后的坐标
    feats = torch.ones((len(coords),1)).float()
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=x.device)
    return x

# 定义一个函数，用于根据坐标对稀疏张量进行排序
def sort_sparse_tensor_with_dir(sparse_tensor, dir=1):
    """
    Sort points in sparse tensor according to their coordinates.
    """
    # 计算一个向量，该向量是sparse_tensor的坐标的加权和，其中权重是坐标的最大值加1的幂
    vec = sum([sparse_tensor.C.long().cpu()[:, i] * 
        (sparse_tensor.C.cpu().max().long() + 1) ** (i if dir==0 else (3 - i)) 
        for i in range(sparse_tensor.C.shape[-1])])
    # 根据这个向量对sparse_tensor的特征和坐标进行排序
    indices_sort = np.argsort(vec)
    # 返回一个新的稀疏张量，其中的特征和坐标是排序后的特征和坐标
    sparse_tensor_sort = ME.SparseTensor(features=sparse_tensor.F[indices_sort], 
                                         coordinates=sparse_tensor.C[indices_sort],
                                         tensor_stride=sparse_tensor.tensor_stride[0], 
                                         device=sparse_tensor.device)
    return sparse_tensor_sort



def slice_sparse_tensor(x, slice):
    '''
    slice_sparse_tensor函数用于对稀疏张量进行切片，最多可以获取8个切片。它接受一个稀疏张量x和一个切片数slice。
    如果slice为0，那么直接返回x。否则，它首先计算x的坐标的方差和中位数，然后根据方差对坐标进行排序。
    接着，根据slice的值，生成一系列的掩码，然后使用这些掩码对x进行切片，最后返回切片后的稀疏张量列表。
    A simple function to slice a sparse tensor, can at most get 8 slices
    '''

    if slice == 0: return [x]
    vars = torch.var(x.C[:, 1:].cpu().float(), dim=0).numpy()
    thres = np.percentile(x.C[:, 1:].cpu().numpy(), 50, axis=0)
    axis_l = [AxisSlice('x', vars[0], thres[0], x.C[:, 1] < thres[0]),
                 AxisSlice('x', vars[1], thres[1], x.C[:,2] < thres[1]),
                 AxisSlice('x', vars[2], thres[2], x.C[:,3] < thres[2])]
    axis_l = sorted(axis_l, key=lambda axis: axis.var, reverse=True)

    x_list = []
    if slice == 1:
        masks = [
            axis_l[0].mask,
            axis_l[0].nm(),
        ]
    elif slice == 2:
        masks = [
            torch.logical_and(axis_l[0].mask, axis_l[1].mask),
            torch.logical_and(axis_l[0].nm(), axis_l[1].mask),
            torch.logical_and(axis_l[0].mask, axis_l[1].nm()),
            torch.logical_and(axis_l[0].nm(), axis_l[1].nm())
        ]
    elif slice == 3:
        masks = [
            torch.logical_and(torch.logical_and(axis_l[0].mask, axis_l[1].mask), axis_l[2].mask),
            torch.logical_and(torch.logical_and(axis_l[0].nm(), axis_l[1].mask), axis_l[2].mask),
            torch.logical_and(torch.logical_and(axis_l[0].mask, axis_l[1].nm()), axis_l[2].mask),
            torch.logical_and(torch.logical_and(axis_l[0].nm(), axis_l[1].nm()), axis_l[2].mask),
            torch.logical_and(torch.logical_and(axis_l[0].mask, axis_l[1].mask), axis_l[2].nm()),
            torch.logical_and(torch.logical_and(axis_l[0].nm(), axis_l[1].mask), axis_l[2].nm()),
            torch.logical_and(torch.logical_and(axis_l[0].mask, axis_l[1].nm()), axis_l[2].nm()),
            torch.logical_and(torch.logical_and(axis_l[0].nm(), axis_l[1].nm()), axis_l[2].nm())
        ]

    for mask in masks:
        x_list.append(ME.SparseTensor(
                        features=torch.ones((torch.sum(mask), 1)).float(), 
                        coordinates=x.C[mask], 
                        tensor_stride=1, device=x.device))
    return x_list


class AxisSlice:
    def __init__(self, name, var, thres, mask):
        self.name = name
        self.var = var
        self.thres = thres
        self.mask = mask

    def __repr__(self):
        return repr((self.name, self.var, self.thres, self.mask))

    def nm(self):
        return torch.logical_not(self.mask)
    # AxisSlice类用于存储坐标的信息。它接受一个名称name，一个方差var，一个阈值thres和一个掩码mask，
    # 然后将这些信息存储为类的属性。
