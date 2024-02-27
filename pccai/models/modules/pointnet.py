# 导入所需的库
import torch
import torch.nn as nn
from pccai.models.utils import PointwiseMLP, GlobalPool
from torch_scatter import scatter_max, scatter_min, scatter_mean

# 定义PointNet类，它是一个在同质批处理模式下工作的PointNet模型
class PointNet(nn.Module):
    """PointNet模型在同质批处理模式下工作。

    参数:
        mlp_dims: MLP的维度
        fc_dims: 在最大池化后的FC的维度
        mlp_dolastrelu: 是否在MLP后做最后的ReLu
    """

    def __init__(self, net_config, **kwargs):
        super(PointNet, self).__init__()
        # 创建一个PointwiseMLP对象，它是可学习的
        self.pointwise_mlp = PointwiseMLP(net_config['mlp_dims'], net_config.get('mlp_dolastrelu', False))
        # 创建一个FC对象，它也是可学习的
        self.fc = PointwiseMLP(net_config['fc_dims'], net_config.get('fc_dolastrelu', False))

        # 创建一个全局池化对象，它使用自适应最大池化
        self.global_pool = GlobalPool(nn.AdaptiveMaxPool2d((1, net_config['mlp_dims'][-1])))

    def forward(self, data):
        # 将输入数据首先传递给PointwiseMLP对象进行处理，然后进行全局池化，最后传递给FC对象进行处理
        return self.fc(self.global_pool(self.pointwise_mlp(data)))

class PointNetHetero(nn.Module):
    """PointNet在异质批处理模式下工作。"""

    def __init__(self, net_config, **kwargs):
        super(PointNetHetero, self).__init__()
        # 创建一个PointwiseMLP对象，它是可学习的
        self.pointwise_mlp = PointwiseMLP(net_config['mlp_dims'], net_config.get('mlp_dolastrelu', False))
        # 创建一个FC对象，它也是可学习的
        self.fc = PointwiseMLP(net_config['fc_dims'], False)
        # 从net_config中获取ext_cw的值
        self.ext_cw = net_config.get('ext_cw', False)

        # 从kwargs中获取syntax对象，并从中提取syntax_gt和syntax_cw
        self.syntax_gt = kwargs['syntax'].syntax_gt
        self.syntax_cw = kwargs['syntax'].syntax_cw

    def forward(self, data):
        # 获取输入数据data的设备信息
        device = data.device

        # 获取输入数据data的批量大小、点数和维度
        batch_size, pnt_cnt, dims = data.shape[0], data.shape[1], data.shape[2]
        # 将输入数据data重新塑形为一维
        data = data.view(-1, dims)
        # 使用cumsum()计算块索引
        block_idx = torch.cumsum(data[:, self.syntax_gt['block_start']] > 0, dim=0) - 1
        # 删除填充和跳过的点
        block_idx = block_idx[data[:, self.syntax_gt['block_pntcnt']] > 0]
        # 为每个点云的开始创建一个索引
        pc_start = torch.arange(0, batch_size, dtype=torch.long, device=device).repeat_interleave(pnt_cnt)
        # 删除"非开始"点
        pc_start = pc_start[data[:, self.syntax_gt['block_start']] > 0]
        # 创建一个新的张量pc_start
        pc_start = torch.cat((torch.ones(1, device=device), pc_start[1:] - pc_start[0: -1]))
        # 删除填充和跳过的点
        data = data[data[:, self.syntax_gt['block_pntcnt']] > 0, :]

        # 归一化点云：平移和缩放
        xyz_slc = slice(self.syntax_gt['xyz'][0], self.syntax_gt['xyz'][1] + 1)
        data[:, xyz_slc] -= data[:, self.syntax_gt['block_center'][0] : self.syntax_gt['block_center'][1] + 1]
        data[:, xyz_slc] *= data[:, self.syntax_gt['block_scale']].unsqueeze(-1)

        # 获取3D点
        pnts_3d = data[:, xyz_slc]
        # 在这种情况下，使用xyz坐标作为特征
        point_feature = self.pointwise_mlp(pnts_3d)
        # 如果ext_cw为True，则将三个特征（最大值、最小值、平均值）连接在一起
        if self.ext_cw:
            cw_inp1 = scatter_max(point_feature, block_idx.long(), dim=0)[0]
            cw_inp2 = scatter_min(point_feature, block_idx.long(), dim=0)[0]
            cw_inp3 = scatter_mean(point_feature, block_idx.long(), dim=0)
            cw_inp = torch.cat([cw_inp1, cw_inp2, cw_inp3], dim=1)
        else:
            # 否则，只使用最大值特征
            cw_inp = scatter_max(point_feature, block_idx.long(), dim=0)[0]
        # 获取块特征
        block_feature = self.fc(cw_inp)
        # 创建一个掩码，用于标记大于0的块开始点
        mask = data[:, self.syntax_gt['block_start']] > 0

        # 返回带有元数据的codeword
        out = torch.zeros(torch.sum(mask), self.syntax_cw['__len__'], device=device)
        out[:, self.syntax_cw['cw'][0] : self.syntax_cw['cw'][1] + 1] = block_feature
        out[:, self.syntax_cw['block_pntcnt']] = data[mask, self.syntax_gt['block_pntcnt']]
        out[:, self.syntax_cw['block_center'][0] : self.syntax_cw['block_center'][1] + 1] = data[mask, self.syntax_gt['block_center'][0] : self.syntax_gt['block_center'][1] + 1]
        out[:, self.syntax_cw['block_scale']] = data[mask, self.syntax_gt['block_scale']]
        out[:, self.syntax_cw['pc_start']] = pc_start
        return out
