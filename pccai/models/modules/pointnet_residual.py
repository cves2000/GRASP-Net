# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Geometric subtraction and point analysis in the GRASP-Net paper

# 导入所需的库
import os, sys
import torch
import torch.nn as nn
import numpy as np
from pccai.models.modules.pointnet import PointNet

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../third_party/nndistance'))
from modules.nnd import NNDModule
nndistance = NNDModule()

# 尝试导入faiss库，如果没有找到，则将found_FAISS设置为False
try:
    import faiss
    import faiss.contrib.torch_utils
    found_FAISS = True
except ModuleNotFoundError:
    found_FAISS = False

# 定义PointResidualEncoder类
class PointResidualEncoder(nn.Module): #实现了几何减法和点分析

    def __init__(self, net_config, **kwargs):
        super(PointResidualEncoder, self).__init__()

        # 从kwargs中获取语法对象，并获取阶段信息
        syntax = kwargs['syntax']
        self.phase = syntax.phase.lower()
        # 从net_config中获取k和thres_dist的值
        self.k = net_config['k']
        self.thres_dist = net_config['thres_dist']
        # 创建一个PointNet对象，用于特征生成，即点分析部分
        self.feat_gen = PointNet(net_config, syntax=syntax)
        # 检查是否使用faiss库，并检查是否找到faiss库
        self.faiss = (net_config.get('faiss', False) or self.phase != 'train') and found_FAISS
        self.faiss_resource, self.faiss_gpu_index_flat = None, None
        self.faiss_exact_search = True

    def forward(self, x_orig, x_coarse):
        # 根据阶段选择适当的几何减法函数
        geo_subtraction = self.geo_subtraction_batch if self.phase =='train' else self.geo_subtraction
        # 执行几何减法
        geo_res = geo_subtraction(x_orig, x_coarse)
        # 生成特征
        feat = self.feat_gen(geo_res)
        return feat

    # This is to perform geometric subtraction for point clouds in a batch manner 
  # 这个函数用于在批处理模式下对点云进行几何减法
    def geo_subtraction_batch(self, x_orig, x_coarse):
    
        # 初始化几何残差
        geo_res = torch.zeros(size=(x_coarse.shape[0], self.k, 3), device=x_coarse.device)
        # 获取批量大小
        batch_size = x_orig[-1][0].item() + 1
        tot = 0
    
        # 对每个点云进行操作
        for pc_cnt in range(batch_size):
    
            # 如果使用FAISS进行一次性kNN搜索
            if self.faiss == True:
                # 获取当前粗糙和完整的点云
                x_coarse_cur = (x_coarse[x_coarse[:, 0] == pc_cnt][:, 1:]).float().contiguous()
                x_orig_cur = (x_orig[x_orig[:, 0] == pc_cnt][:, 1:]).float().contiguous()
                # 如果没有FAISS GPU索引，则创建一个
                if self.faiss_gpu_index_flat == None:
                    self.faiss_resource = faiss.StandardGpuResources()
                    self.faiss_gpu_index_flat = faiss.GpuIndexFlatL2(self.faiss_resource, 3)
                # 添加原始点云到FAISS GPU索引
                self.faiss_gpu_index_flat.add(x_orig_cur)
                # 实际搜索
                _, I = self.faiss_gpu_index_flat.search(x_coarse_cur, self.k)
                # 重置FAISS GPU索引
                self.faiss_gpu_index_flat.reset()
                # 重复粗糙点云
                x_coarse_rep = x_coarse_cur.unsqueeze(1).repeat_interleave(self.k, dim=1)
                # 计算几何残差
                geo_res[tot : tot + x_coarse_cur.shape[0], :, :] = x_orig_cur[I] - x_coarse_rep
    
                # 去除离群值
                mask = torch.logical_or(
                    torch.max(geo_res[tot : tot + x_coarse_cur.shape[0], :, :], dim=2)[0] > self.thres_dist,
                    torch.min(geo_res[tot : tot + x_coarse_cur.shape[0], :, :], dim=2)[0] < -self.thres_dist
                ) # True表示离群值
                # 获取第一个最近邻的索引
                I[mask] = I[:, 0].unsqueeze(-1).repeat_interleave(self.k, dim=1)[mask]
                # 重新计算离群值的距离
                geo_res[tot : tot + x_coarse_cur.shape[0], :, :][mask] = x_orig_cur[I[mask]] - x_coarse_rep[mask]
                tot += x_coarse_cur.shape[0]
    
            else: # 使用nndistance进行顺序最近邻搜索
                # 获取当前粗糙和完整的点云
                x_coarse_cur = (x_coarse[x_coarse[:, 0] == pc_cnt][:, 1:]).float().unsqueeze(0).contiguous()
                x_orig_cur = (x_orig[x_orig[:, 0] == pc_cnt][:, 1:]).float().unsqueeze(0).contiguous()
                for nn_cnt in range(self.k):
                    if x_orig_cur.shape[1] > 0:
                        # 计算最近邻
                        _, _, idx_coarse, _ = nndistance(x_coarse_cur, x_orig_cur)
                        # 计算几何残差
                        geo_res[tot : tot + x_coarse_cur.shape[1], nn_cnt, :] = x_orig_cur.squeeze(0)[idx_coarse] - x_coarse_cur.squeeze(0)
                        # 创建一个掩码，False表示离群值
                        mask = torch.logical_and(
                            torch.max(geo_res[tot : tot + x_coarse_cur.shape[1], nn_cnt, :], dim=1)[0] <= self.thres_dist,
                            torch.min(geo_res[tot : tot + x_coarse_cur.shape[1], nn_cnt, :], dim=1)[0] >= -self.thres_dist
                         )
     
                        # 获取离群值的序列
                        seq_outlier = torch.arange(tot, x_coarse_cur.shape[1] + tot)[torch.logical_not(mask)]
                        # 从最近邻集合中移除离群值
                        geo_res[seq_outlier, nn_cnt, :] = geo_res[seq_outlier, nn_cnt - 1, :]
                        # 从最近邻集合中移除离群值
                        idx_coarse = idx_coarse[mask.unsqueeze(0)]
                        # 创建一个掩码，用于获取剩余的点
                        mask = torch.ones(x_orig_cur.shape[1], dtype=bool, device=x_orig.device)
                        mask[idx_coarse.squeeze(0)] = False
                        # 获取剩余的点
                        x_orig_cur = x_orig_cur[mask.unsqueeze(0)].unsqueeze(0)
                    else: # 如果没有剩余的点，复制最后一个
                        geo_res[tot : tot + x_coarse_cur.shape[1], nn_cnt:, :] = \
                            geo_res[tot : tot + x_coarse_cur.shape[1], nn_cnt - 1, :].unsqueeze(1)
                        break
                tot += x_coarse_cur.shape[1]
        return geo_res



    # 这个函数用于对一个点云进行几何减法，主要在推理过程中使用
    def geo_subtraction(self, x_orig, x_coarse):
        # 初始化几何残差
        geo_res = torch.zeros(size=(x_coarse.shape[1], self.k, 3), device=x_coarse.device)
        # 去掉x_orig和x_coarse的第一个维度
        x_orig, x_coarse = x_orig.squeeze(0), x_coarse.squeeze(0)
        # 创建一个FAISS资源
        self.faiss_resource = faiss.StandardGpuResources()
    
        # 执行kNN搜索
        if self.faiss_exact_search: # 精确搜索
            # 创建一个FAISS GPU索引
            self.faiss_gpu_index_flat = faiss.GpuIndexFlatL2(self.faiss_resource, 3)
            # 将原始点云添加到FAISS GPU索引
            self.faiss_gpu_index_flat.add(x_orig)
            # 一次性搜索
            _, I = self.faiss_gpu_index_flat.search(x_coarse, self.k)
        else: # 近似搜索
            # 创建一个FAISS GPU索引
            self.faiss_gpu_index_flat = faiss.GpuIndexIVFFlat(self.faiss_resource, 3, 4 * int(np.ceil(np.sqrt(x_orig.shape[0]))), faiss.METRIC_L2)
            # 训练FAISS GPU索引
            self.faiss_gpu_index_flat.train(x_orig)
            # 将原始点云添加到FAISS GPU索引
            self.faiss_gpu_index_flat.add(x_orig)
            # 初始化索引
            I = torch.zeros(x_coarse.shape[0], self.k, device=x_coarse.device, dtype=torch.long)
            max_query = 2 ** 16
            n_times = int(np.ceil(x_coarse.shape[0] / max_query))
            # 由于GpuIndexIVFFlat的限制，需要通过批处理进行搜索
            for cnt in range(n_times):
                slc = slice(cnt * max_query, x_coarse.shape[0] if cnt == n_times -1 else (cnt + 1) * max_query - 1)
                I[slc, :] = self.faiss_gpu_index_flat.search(x_coarse[slc, :], self.k)[1]
    
        # 重置FAISS GPU索引
        self.faiss_gpu_index_flat.reset()
        # 重复粗糙点云
        x_coarse_rep = x_coarse.unsqueeze(1).repeat_interleave(self.k, dim=1)
        # 计算几何残差
        geo_res = x_orig[I] - x_coarse_rep
    
        # 去除离群值
        mask = torch.logical_not(torch.logical_and(
            torch.max(geo_res, dim=2)[0] <= self.thres_dist,
            torch.min(geo_res, dim=2)[0] >= -self.thres_dist
        )) # True表示离群值
        # 获取第一个最近邻的索引
        I[mask] = I[:, 0].unsqueeze(-1).repeat_interleave(self.k, dim=1)[mask]
        # 重新计算离群值的距离
        geo_res[mask] = x_orig[I[mask]] - x_coarse_rep[mask]
        # 删除不再需要的变量以节省内存
        del I, x_coarse_rep, x_orig, x_coarse, mask
        return geo_res
