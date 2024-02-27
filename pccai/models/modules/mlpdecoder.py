# 导入所需的库
import torch
import torch.nn as nn
import numpy as np
from pccai.models.utils import PointwiseMLP

# 定义MlpDecoder类，它是一个在同质批处理模式下工作的MLP解码器
class MlpDecoder(nn.Module):
    def __init__(self, net_config, **kwargs):
        super(MlpDecoder, self).__init__()
        # 从net_config中获取num_points和dims的值
        self.num_points = net_config['num_points']
        dims = net_config['dims']
        # 创建一个PointwiseMLP对象
        self.mlp = PointwiseMLP(dims + [3 * self.num_points], doLastRelu=False)

    def forward(self, cw):
        # 将输入数据cw传递给PointwiseMLP对象进行处理
        out1 = self.mlp(cw) 
        # 将处理后的数据重新塑形并返回
        return out1.view(cw.shape[0], self.num_points, -1)

# 定义MlpDecoderHetero类，它是一个在异质批处理模式下工作的MLP解码器
class MlpDecoderHetero(nn.Module):
    def __init__(self, net_config, **kwargs):
        super(MlpDecoderHetero, self).__init__()
        # 从net_config中获取num_points和dims的值
        self.num_points = net_config['num_points']
        dims = net_config['dims']
        # 创建一个PointwiseMLP对象
        self.mlp = PointwiseMLP(dims + [3 * self.num_points], doLastRelu=False)

        # 从kwargs中获取syntax对象，并从中提取syntax_cw和syntax_rec
        self.syntax_cw = kwargs['syntax'].syntax_cw
        self.syntax_rec = kwargs['syntax'].syntax_rec

    def forward(self, cw):
        # 获取输入数据cw的设备信息
        device = cw.device
        # 将cw的一部分数据传递给PointwiseMLP对象进行处理
        pc_block = self.mlp(cw[:, self.syntax_cw['cw'][0] : self.syntax_cw['cw'][1] + 1])
        pc_block = pc_block.view(cw.shape[0] * self.num_points, -1)

        # 创建新的张量
        block_npts = torch.ones(cw.shape[0], dtype=torch.long, device=device) * self.num_points
        # 为每个点指定其codeword/block的索引
        cw_idx = torch.arange(block_npts.shape[0], device=device).repeat_interleave(block_npts)
        # 如果一个点是一个block的第一个点，则标记为1
        block_start = torch.cat((torch.ones(1, device=device), cw_idx[1:] - cw_idx[:-1])).float()

        # 反归一化点云
        center = cw[:, self.syntax_cw['block_center'][0]: self.syntax_cw['block_center'][1] + 1].repeat_interleave(block_npts, 0)
        scale = cw[:, self.syntax_cw['block_scale']: self.syntax_cw['block_scale'] + 1].repeat_interleave(block_npts, 0)

        # 从cw中的pc_start（blocks）构建points的pc_start
        pc_start = torch.zeros(cw.shape[0], device=device).repeat_interleave(block_npts)
        # 为每个block指定起始点索引
        block_idx = torch.cat((torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(block_npts, 0)[:-1]), 0)
        # 如果一个点是其点云的第一个，则标记为1
        pc_start[block_idx] = cw[:, self.syntax_cw['pc_start']: self.syntax_cw['pc_start'] + 1].squeeze(-1)

        # 反归一化：缩放和平移
        pc_block = pc_block / scale # 缩放
        pc_block = pc_block + center # 平移

        # 组装输出
        out = torch.zeros(pc_block.shape[0], self.syntax_rec['__len__']).cuda()
        out[:, self.syntax_rec['xyz'][0] : self.syntax_rec['xyz'][1] + 1] = pc_block
        out[:, self.syntax_rec['block_start']] = block_start
        out[:, self.syntax_rec['block_center'][0] : self.syntax_rec['block_center'][1] + 1] = center
        out[:, self.syntax_rec['block_scale']] = scale
        out[:, self.syntax_rec['pc_start']] = pc_start
        return out


def prepare_meta_data(self, binstrs, block_pntcnt, octree_organizer):
        """将八叉树的二进制字符串转换为叶节点的一组比例和中心，然后根据解码的语法将它们组织为元数据数组。"""

        # 将八叉树字符串分区为块
        leaf_blocks = octree_organizer.departition_octree(binstrs, block_pntcnt)
        # 初始化元数据数组
        meta_data = np.zeros((len(leaf_blocks), self.syntax_cw['__len__'] - self.syntax_cw['__meta_idx__']), dtype=np.float32)
        cur = 0

        # 组装元数据
        meta_data[0, self.syntax_cw['pc_start'] - self.syntax_cw['__meta_idx__']] = 1
        for idx, block in enumerate(leaf_blocks):
            if block['binstr'] >= 0: # 只保留具有变换模式的块
                # 获取块的中心和规模
                center, scale = octree_organizer.get_normalizer(block['bbox_min'], block['bbox_max'])
                # 将块的点数、规模和中心添加到元数据中
                meta_data[cur, self.syntax_cw['block_pntcnt'] - self.syntax_cw['__meta_idx__']] = block_pntcnt[idx]
                meta_data[cur, self.syntax_cw['block_scale'] - self.syntax_cw['__meta_idx__']] = scale
                meta_data[cur, self.syntax_cw['block_center'][0] - self.syntax_cw['__meta_idx__'] : 
                    self.syntax_cw['block_center'][1] - self.syntax_cw['__meta_idx__'] + 1] = center
                cur += 1
        return meta_data

        # Only returns the useful part
        return torch.as_tensor(meta_data[:cur, :], device=torch.device('cuda')).unsqueeze(-1).unsqueeze(-1)
        # meta_data[:cur, :]：这部分代码获取元数据数组的前cur行。cur是在前面的循环中计算出来的，表示有用的数据的数量。
        # torch.as_tensor(..., device=torch.device('cuda'))：这部分代码将获取到的数据转换为PyTorch张量，并将其放在CUDA设备上。这样可以利用GPU进行高效的计算。
        # .unsqueeze(-1).unsqueeze(-1)：这部分代码在张量的最后两个维度上增加新的维度。这通常是为了满足某些特定的数据形状要求。
