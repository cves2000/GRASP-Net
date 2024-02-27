# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.

# GRASP-Net: Geometric Residual Analysis and Synthesis for Point Cloud Compression
# 导入所需的库
import os, sys
import torch
import torch.nn as nn
import numpy as np
import time
import MinkowskiEngine as ME

# 导入其他模块和函数
from pccai.models.modules.get_modules import get_module_class
from pccai.models.utils_sparse import scale_sparse_tensor_batch, sort_sparse_tensor_with_dir
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../third_party/PCGCv2'))
from data_utils import read_ply_ascii_geo, write_ply_ascii_geo
from entropy_model import EntropyBottleneck
from gpcc import gpcc_encode, gpcc_decode
from data_utils import scale_sparse_tensor

# 定义一个类，用于进行点云压缩的几何残差分析和合成
class GeoResCompression(nn.Module):
    def __init__(self, net_config, syntax):
        super(GeoResCompression, self).__init__()

        # 获取基本参数
        self.dus = net_config.get('dus', 1) # 下上缩放可以是1或2
        self.scaling_ratio = net_config['scaling_ratio']
        self.eb_channel = net_config['entropy_bottleneck']
        self.entropy_bottleneck = EntropyBottleneck(self.eb_channel)
        self.thres_dist = np.ceil((1 / self.scaling_ratio) * 0.65) if self.scaling_ratio < 0.5 else 1

        self.point_mul = net_config.get('point_mul', 5)
        self.skip_mode = net_config.get('skip_mode', False)
        if syntax.phase.lower() == 'train': # 添加噪声，只在训练时存在
            self.noise = net_config.get('noise', -1)

        # 覆盖/填充一些子模块的参数
        net_config['res_enc']['k'] =  net_config['res_dec']['num_points'] = self.point_mul
        net_config['res_enc']['thres_dist'] = self.thres_dist
        net_config['res_dec']['dims'][0] = net_config['vox_dec']['dims'][-1]

        # 构造网络模块
        self.res_dec = get_module_class(net_config['res_dec']['model'], False)(net_config['res_dec'], syntax=syntax)
        self.vox_dec = get_module_class(net_config['vox_dec']['model'], False)(net_config['vox_dec'], syntax=syntax)
        self.pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        if self.skip_mode == False:
            self.vox_enc = get_module_class(net_config['vox_enc']['model'], False)(net_config['vox_enc'], syntax=syntax)
            self.res_enc = get_module_class(net_config['res_enc']['model'], False)(net_config['res_enc'], syntax=syntax)

    def forward(self, coords):
        # 从稀疏张量构造坐标
        coords[0][0] = 0
        coords[:, 0] = torch.cumsum(coords[:,0], 0)
        device = coords.device
        x = ME.SparseTensor(
            features=torch.ones(coords.shape[0], 1, device=device, dtype=torch.float32),
            coordinates=coords, 
            device=device)
    # 这段代码主要定义了一个名为GeoResCompression的类，该类用于进行点云压缩的几何残差分析和合成。在这个类中，首先获取了一些基本参数，然后覆盖或填充了一些子模块的参数，接着构造了网络模块，最后定义了前向传播函数。在前向传播函数中，它从稀疏张量构造了坐标。这个类是GRASP-Net模型的一部分，GRASP-Net模型是一种用于点云压缩的模型。
    # dus：这是一个参数，表示下上缩放，可以是1或2。
    # scaling_ratio：这是一个参数，表示缩放比例。
    # eb_channel：这是一个参数，表示熵瓶颈的通道数。
    # entropy_bottleneck：这是一个熵瓶颈对象，用于进行熵编码和解码。
    # thres_dist：这是一个参数，表示阈值距离，用于确定点云的采样密度。
    # point_mul：这是一个参数，表示点的倍数，用于确定每个点的邻居数量。
    # skip_mode：这是一个参数，表示是否跳过某些模式。
    # noise：这是一个参数，表示噪声，只在训练时存在。
    # res_dec：这是一个解码器对象，用于进行残差解码。
    # vox_dec：这是一个解码器对象，用于进行体素解码。
    # pool：这是一个池化对象，用于进行最大池化。
    # vox_enc：这是一个编码器对象，用于进行体素编码。
    # res_enc：这是一个编码器对象，用于进行残差编码。


    # 用于模拟基层
    with torch.no_grad():
        # 量化
        x_coarse = scale_sparse_tensor_batch(x, factor=self.scaling_ratio)
        x_coarse = sort_sparse_tensor_with_dir(x_coarse)
        # 这里应该无损编码x_coarse，然后进行反量化
        x_coarse_deq = torch.hstack((x_coarse.C[:, 0:1], (x_coarse.C[:, 1:] / self.scaling_ratio)))
    
    # 增强层开始
    if self.skip_mode==False:
        feat = self.res_enc(x.C, x_coarse_deq) # 提取几何残差并进行编码
        x_feat = ME.SparseTensor( # 附加几何特征的粗糙点云
            features=feat,
            coordinate_manager=x_coarse.coordinate_manager,
            coordinate_map_key=x_coarse.coordinate_map_key)
        y = self.vox_enc(x_feat) # 特征编码器
        y_q, likelihood = get_likelihood(self.entropy_bottleneck, y)
    
    else: # 跳过模式
        x_coarse_ds = self.pool(x_coarse if self.dus == 1 else self.pool(x_coarse)) 
        ds_shape = torch.Size((x_coarse_ds.shape[0], self.eb_channel))
    
        y_q = ME.SparseTensor( # 用于上采样的合成特征
            features=torch.ones(ds_shape, device=device),
            coordinate_manager=x_coarse_ds.coordinate_manager,
            coordinate_map_key=x_coarse_ds.coordinate_map_key
        )
        likelihood = torch.ones(ds_shape, device=device)
    
    # 解码器
    res = self.vox_dec(y_q, x_coarse) # 特征解码器
    res = self.res_dec(res) # 特征到残差转换器
    res = sort_sparse_tensor_with_dir(res).F
    res = res.reshape([res.shape[0] * self.point_mul, 3])
    if self.noise > 0: # 添加均匀噪声以增强鲁棒性
        res += (torch.rand(res.shape, device=device) - 0.5) * self.noise
    
    # 添加回残差
    out = x_coarse_deq.repeat_interleave(self.point_mul, dim=0)
    out[:, 1:] += res
    return {'x_hat': out,
            'gt': coords,
            'likelihoods': {'feats': likelihood}}
    # 在这段代码中，首先进行了量化和反量化操作，然后根据是否跳过模式进行了不同的处理。
    # 在非跳过模式下，提取了几何残差并进行了编码，然后对特征进行了编码，并获取了可能性。
    # 在跳过模式下，进行了上采样的特征合成。然后，对特征进行了解码，将特征转换为残差，如果存在噪声，则添加了均匀噪声以增强鲁棒性。
    # 最后，将残差添加回去，得到了输出。


      def compress(self, x, tag):
        """
        这个函数使用学习到的熵瓶颈的统计信息进行实际的压缩，一次消耗一个点云。
        """
    
        # 开始压缩
        x_coarse = scale_sparse_tensor(x, factor=self.scaling_ratio) # 对x进行缩放
        filename_base = tag + '_B.bin' # 定义基础文件名
        start = time.monotonic() # 获取开始时间
        coord_codec(filename_base, x_coarse.C.detach().cpu()[:, 1:]) # 使用G-PCC无损编码
        base_enc_time = time.monotonic() - start # 计算编码时间
        if self.skip_mode: # 处理跳过模式
            string, min_v, max_v, shape = None, None, None, None
            del x
            torch.cuda.empty_cache()
        else:
            x_coarse_deq = (x_coarse.C[:, 1:] / self.scaling_ratio).float().unsqueeze(0).contiguous() # 反量化
            x_c = x.C[:, 1:].float().unsqueeze(0)
            del x
            torch.cuda.empty_cache()
            feat = self.res_enc(x_c, x_coarse_deq) # 提取几何残差并进行编码
    
            # 使用feat构建低粗糙度的点云
            x_feat = ME.SparseTensor( # 低位深度PC与attr
                    features=feat,
                    coordinate_manager=x_coarse.coordinate_manager,
                    coordinate_map_key=x_coarse.coordinate_map_key)
    
            y = self.vox_enc(x_feat) # 体素编码器
            y = sort_sparse_tensor_with_dir(y)
            shape = y.F.shape
            string, min_v, max_v = self.entropy_bottleneck.compress(y.F.cpu()) # 使用熵瓶颈进行压缩
    
        return filename_base, [string], [min_v], [max_v], [shape], x_coarse.shape[0], base_enc_time
    #     在这段代码中，首先对输入的点云x进行了缩放，然后使用G-PCC进行了无损编码。接着，根据是否跳过模式进行了不同的处理。在非跳过模式下，进行了反量化操作，
    # 提取了几何残差并进行了编码，然后构建了低粗糙度的点云，并进行了体素编码，最后使用熵瓶颈进行了压缩。
    # 在跳过模式下，直接删除了x并清空了CUDA缓存。最后，返回了基础文件名、字符串、最小值、最大值、形状、x_coarse的形状和基础编码时间。

def decompress(self, filename_base, string, min_v, max_v, shape, base_dec_time):
    """
    这个函数使用学习到的熵瓶颈的统计信息进行实际的解压缩，一次消耗一个点云。
    """
    start = time.monotonic() # 获取开始时间
    y_C = coord_codec(filename_base) # 使用G-PCC无损解码
    base_dec_time[0] = time.monotonic() - start # 计算解码时间
    y_C = torch.cat((torch.zeros((len(y_C), 1)).int(), torch.tensor(y_C).int()), dim=-1) # 拼接张量
    if self.skip_mode and self.base_only: # 处理跳过模式和仅基础模式
        y_C = (y_C[:, 1:] / self.scaling_ratio).float().contiguous() # 反量化
        return y_C

    # 从y_C创建用于解码的y_C的下采样版本
    device = next(self.parameters()).device # 获取设备
    y_dummy = ME.SparseTensor( # 创建稀疏张量
            features=torch.ones((y_C.shape[0], 1), device=device),
            coordinates=y_C, 
            tensor_stride=1, 
            device=device
        )
    del y_C
    torch.cuda.empty_cache() # 清空CUDA缓存

    y_ds = self.pool(y_dummy if self.dus == 1 else self.pool(y_dummy)) 
    if self.skip_mode: # 超分辨率
        y_down = ME.SparseTensor(
                features=torch.ones((y_ds.shape[0], self.eb_channel), device=device),
                coordinate_manager=y_ds.coordinate_manager,
                coordinate_map_key=y_ds.coordinate_map_key
            )
    else:
        y_ds = sort_sparse_tensor_with_dir(y_ds)
        y_F = self.entropy_bottleneck.decompress(string[0], min_v[0], max_v[0], shape[0], channels=shape[0][-1]) # 使用熵瓶颈进行解压缩
        y_down = ME.SparseTensor(features=y_F, device=device,
            coordinate_manager=y_ds.coordinate_manager,
            coordinate_map_key=y_ds.coordinate_map_key)

    y_dec = self.vox_dec(y_down, y_dummy) # 特征解码器
    y_dec = self.res_dec(y_dec)
    y_dec_F = y_dec.F.reshape(-1, 3) # 取出解码后的坐标
    y_dec_C = (y_dec.C[:, 1:] / self.scaling_ratio).float().contiguous() # 反量化

    out = y_dec_C.repeat_interleave(self.point_mul, dim=0) + y_dec_F # 添加回残差
    return out
    # 在这段代码中，首先进行了无损解码操作，然后根据是否跳过模式和是否仅基础模式进行了不同的处理。在非跳过模式和非仅基础模式下，
    # 创建了用于解码的y_C的下采样版本，然后进行了反量化操作，使用熵瓶颈进行了解压缩，然后进行了特征解码，取出了解码后的坐标，
    # 进行了反量化操作，最后添加回了残差，得到了输出。


def coord_codec(bin_filename, coords=None):
    ply_filename = bin_filename + '.ply'
    if coords == None: # decode
        gpcc_decode(bin_filename, ply_filename)
        out = read_ply_ascii_geo(ply_filename)
    else: # encode
        coords = coords.numpy().astype('int')
        write_ply_ascii_geo(filedir=ply_filename, coords=coords)
        gpcc_encode(ply_filename, bin_filename)
        out = bin_filename
    os.system('rm '+ ply_filename)
    return out
    # coord_codec函数用于对坐标进行编码和解码。它接受一个二进制文件名bin_filename和一个可选的坐标coords。
    # 如果coords为None，则进行解码操作；否则进行编码操作。解码操作首先使用gpcc_decode函数对二进制文件进行解码，然后使用read_ply_ascii_geo函数读取解码后的.ply文件。
    # 编码操作首先将coords转换为整数类型，然后使用write_ply_ascii_geo函数写入.ply文件，
    # 最后使用gpcc_encode函数对.ply文件进行编码。无论是编码还是解码，最后都会删除.ply文件。

def get_likelihood(entropy_bottleneck, data):
    data_F, likelihood = entropy_bottleneck(data.F, quantize_mode="noise")
    data_Q = ME.SparseTensor(
        features=data_F, 
        coordinate_map_key=data.coordinate_map_key, 
        coordinate_manager=data.coordinate_manager,
        device=data.device)
    return data_Q, likelihood
    # get_likelihood函数用于获取数据的可能性。它接受一个熵瓶颈entropy_bottleneck和一个数据data。
    # 首先，使用entropy_bottleneck的compress方法对数据的特征进行压缩，得到压缩后的特征和可能性。
    # 然后，使用ME.SparseTensor创建一个稀疏张量，其中的特征是压缩后的特征。最后，返回创建的稀疏张量和可能性。
