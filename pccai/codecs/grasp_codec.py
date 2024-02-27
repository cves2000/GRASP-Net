# 关于GRASP-Net编解码器的定义
# 版权所有 (c) 2010-2022，InterDigital
# 保留所有权利。

# 请参阅根文件夹下的LICENSE。

import os  # 导入os库，用于操作系统相关的操作
import sys  # 导入sys库，用于访问与Python解释器相关的变量和函数
import time  # 导入time库，用于处理时间相关的操作
import numpy as np  # 导入numpy库，numpy是Python的一个科学计算的库，提供了矩阵运算的功能

# 需要将其放在这里，以避免与MinkowskiEngine的未知冲突
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party/nndistance'))  # 将第三方库nndistance的路径添加到系统路径中

import torch  # 导入torch库，torch是一个开源的机器学习库，提供了广泛的机器学习算法
import MinkowskiEngine as ME  # 导入MinkowskiEngine库，MinkowskiEngine是一个用于稀疏张量的自动微分库
from pccai.codecs.pcc_codec import PccCodecBase  # 从pccai.codecs.pcc_codec模块导入PccCodecBase类
from pccai.models.utils_sparse import slice_sparse_tensor  # 从pccai.models.utils_sparse模块导入slice_sparse_tensor函数

try:
    import faiss  # 尝试导入faiss库，faiss是一个用于高效相似度搜索和密集向量聚类的库
    found_FAISS = True  # 如果成功导入faiss库，将found_FAISS设置为True
except ModuleNotFoundError:  # 如果导入faiss库失败，将抛出ModuleNotFoundError异常
    found_FAISS = False  # 如果导入faiss库失败，将found_FAISS设置为False

class GeoResCompressionCodec(PccCodecBase):  # 定义一个名为GeoResCompressionCodec的类，该类继承自PccCodecBase类
    """
    Geometric Residual Analysis and Synthesis for PCC, m58962, Jan 22, the codec itself
    """

    def __init__(self, codec_config, pccnet, bit_depth, syntax):  # 类的初始化方法
        super().__init__(codec_config, pccnet, syntax)  # 调用父类的初始化方法
        self.res = 2 ** bit_depth  # 计算2的bit_depth次方，并将结果赋值给self.res
        pccnet.base_only = codec_config.get('base_only', False)  # 从编解码器配置中获取base_only参数，如果没有找到，则默认为False
        if pccnet.skip_mode == False:  # 如果pccnet的skip_mode参数为False
            pccnet.res_enc.faiss = codec_config.get('faiss', True) and found_FAISS == True  # 从编解码器配置中获取faiss参数，如果没有找到，则默认为True，并且如果found_FAISS为True，则将pccnet.res_enc.faiss设置为True

        # 设置切片参数
        self.slice = codec_config.get('slice', 0)  # 从编解码器配置中获取slice参数，如果没有找到，则默认为0

        # 对于表面点云（<=16bit），切片参数被经验性地设置为避免高内存成本
        if bit_depth >= 12 and bit_depth <= 16:  # 如果bit_depth在12到16之间
            if pccnet.scaling_ratio >= 0.625:  # 如果pccnet的scaling_ratio参数大于等于0.625
                self.slice = 3  # 将self.slice设置为3
            elif pccnet.scaling_ratio >= 0.5:  # 如果pccnet的scaling_ratio参数大于等于0.5
                self.slice = 2  # 将self.slice设置为2
            elif pccnet.scaling_ratio >= 0.375:  # 如果pccnet的scaling_ratio参数大于等于0.375


    def compress(self, coords, tag):
        """
        压缩点云中的所有变换块并将比特流写入文件。
        """
    
        # 从点云构造稀疏张量，如果需要的话进行切片
        start = time.monotonic()  # 获取当前时间
        coords = (coords + np.array(self.translate)) * self.scale  # 对坐标进行平移和缩放
        pnt_cnt = coords.shape[0]  # 获取点云的数量
        coords = torch.tensor(coords)  # 将坐标转换为torch张量
        feats = torch.ones((len(coords), 1)).float()  # 创建一个全为1的特征张量
        coords, feats = ME.utils.sparse_collate([coords], [feats])  # 将坐标和特征进行稀疏整理
        device = next(self.pccnet.parameters()).device  # 获取设备信息
        x_list = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)  # 构造稀疏张量
        x_list = slice_sparse_tensor(x_list, self.slice)  # 对稀疏张量进行切片
        end = time.monotonic()  # 获取当前时间
    
        # 准备工作
        filename_list = []  # 创建一个空的文件名列表
        stat_dict = {  # 创建一个统计字典
            'scaled_num_points': 0,
            'all_enc_time': end - start,
            'base_enc_time': 0,
            'bpp_base': 0
        }
        
        for cnt_slice, x in enumerate(x_list):  # 遍历切片后的稀疏张量列表
    
            # 使用网络进行压缩
            start = time.monotonic()  # 获取当前时间
            filename_base, string_set, min_v_set, max_v_set, shape_set, scaled_num_points, base_enc_time = self.pccnet.compress(x, tag + '_' + str(cnt_slice))  # 使用网络进行压缩，并获取结果
            end = time.monotonic()  # 获取当前时间
            stat_dict['scaled_num_points'] += scaled_num_points  # 更新统计字典中的 'scaled_num_points' 值
            stat_dict['all_enc_time'] += end - start  # 更新统计字典中的 'all_enc_time' 值
            stat_dict['base_enc_time'] += base_enc_time  # 更新统计字典中的 'base_enc_time' 值
            stat_dict['bpp_base'] += os.stat(filename_base).st_size * 8 / pnt_cnt  # 更新统计字典中的 'bpp_base' 值
            filename_list.append(filename_base)  # 将基础文件名添加到文件名列表中
            if self.pccnet.skip_mode == False:  # 如果网络的skip_mode参数为False
                filename_enhance = tag + '_' + str(cnt_slice) + '_E' + '.bin'  # 创建增强文件名
                filename_header = tag + '_' + str(cnt_slice) + '_H' + '.bin'  # 创建头文件名
    
                # 写下字符串
                with open(filename_enhance, 'wb') as fout:  # 打开增强文件
                    for cnt, string in enumerate(string_set):  # 遍历字符串集合
                        fout.write(string)  # 将字符串写入文件
                        key = 'bpp_feat'  # 创建键名
                        if cnt >= 1: key += '_res' + str(len(string_set) - cnt - 1)  # 如果计数大于等于1，更新键名
                        if (key in stat_dict) == False: stat_dict[key] = 0  # 如果键名不在统计字典中，将其添加到统计字典中
                        stat_dict[key] += len(string) * 8 / pnt_cnt  # 更新统计字典中的键值
    
                # 写下头部
                with open(filename_header, 'wb') as fout:  # 打开头文件
                    for cnt in range(len(string_set)):  # 遍历字符串集合
                        fout.write(np.array(shape_set[cnt], dtype=np.int32).tobytes())  # 将形状集合写入文件
                        fout.write(np.array(len(min_v_set[cnt]), dtype=np.int8).tobytes())  # 将最小值集合的长度写入文件
                        fout.write(np.array(min_v_set[cnt], dtype=np.float32).tobytes())  # 将最小值集合写入文件
                        fout.write(np.array(max_v_set[cnt], dtype=np.float32).tobytes())  # 将最大值集合写入文件
                        if cnt != len(string_set) - 1:  # 如果计数不等于字符串集合的长度减1
                            fout.write(np.array(len(string_set[cnt]), dtype=np.int32).tobytes())  # 将字符串集合的长度写入文件
    
                filename_list.append(filename_enhance)  # 将增强文件名添加到文件名列表中
                filename_list.append(filename_header)  # 将头文件名添加到文件名列表中


        # 将额外的统计信息转换为字符串以便于记录日志
        stat_dict['scaled_num_points'] = stat_dict['scaled_num_points']  # 更新 'scaled_num_points' 的值
        stat_dict['enc_time'] = round(stat_dict['all_enc_time'] - stat_dict['base_enc_time'], 3)  # 计算编码时间，并保留三位小数
        stat_dict['all_enc_time'] = round(stat_dict['all_enc_time'], 3)  # 更新 'all_enc_time' 的值，并保留三位小数
        stat_dict['base_enc_time'] = round(stat_dict['base_enc_time'], 3)  # 更新 'base_enc_time' 的值，并保留三位小数
        for k, v in stat_dict.items():  # 遍历统计字典
            if k.find('bpp_') == 0:  # 如果键名以 'bpp_' 开头
                stat_dict[k] = round(v, 6)  # 更新键值，并保留六位小数
        return filename_list, stat_dict  # 返回文件名列表和统计字典



    def decompress(self, filename):
        """
        从文件中解压缩点云的所有变换块。
        """
        stat_dict = {
            'all_dec_time': 0,  # 初始化所有解压缩时间为0
            'base_dec_time': 0,  # 初始化基础解压缩时间为0
        }
        for cnt_slice in range(2 ** self.slice):  # 遍历所有的切片
    
            # 读取字符串
            if self.pccnet.skip_mode == False:  # 如果网络的skip_mode参数为False
                with open(filename[cnt_slice * 3 + 1], 'rb') as fin:  # 打开文件
                    string_set = [fin.read()]  # 读取文件内容
                shape_set, min_v_set, max_v_set = [], [], []  # 初始化形状集合、最小值集合和最大值集合
    
                # 读取头部，然后相应地解析字符串
                with open(filename[cnt_slice * 3 + 2], 'rb') as fin:  # 打开文件
                    shape_set.append(np.frombuffer(fin.read(4*2), dtype=np.int32))  # 读取形状并添加到形状集合中
                    len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]  # 读取最小值的长度
                    min_v_set.append(np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0])  # 读取最小值并添加到最小值集合中
                    max_v_set.append(np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0])  # 读取最大值并添加到最大值集合中
    
            else:
                string_set, min_v_set, max_v_set, shape_set = None, None, None, None  # 如果网络的skip_mode参数为True，将字符串集合、最小值集合、最大值集合和形状集合都设置为None
    
            # 执行解压缩
            base_dec_time = [0]  # 初始化基础解压缩时间为0
            start = time.monotonic()  # 获取当前时间
            if cnt_slice == 0:  # 如果切片计数为0
                pc_rec = self.postprocess(  # 对解压缩后的点云进行后处理
                    self.pccnet.decompress(filename[0], string_set, min_v_set, max_v_set, shape_set, base_dec_time)  # 使用网络进行解压缩，并获取结果
                )
            else:
                pc_rec = torch.vstack((  # 将点云和后处理后的点云进行垂直堆叠
                    pc_rec, self.postprocess(  # 对解压缩后的点云进行后处理
                        self.pccnet.decompress(filename[cnt_slice * (1 if self.pccnet.skip_mode else 3)], 
                            string_set, min_v_set, max_v_set, shape_set, base_dec_time))  # 使用网络进行解压缩，并获取结果
                    )
                )
            end = time.monotonic()  # 获取当前时间
    
            stat_dict['all_dec_time'] += end - start  # 更新统计字典中的 'all_dec_time' 值
            stat_dict['base_dec_time'] += base_dec_time[0]  # 更新统计字典中的 'base_dec_time' 值
    
        stat_dict['dec_time'] = round(stat_dict['all_dec_time'] - stat_dict['base_dec_time'], 3)  # 计算解压缩时间，并更新统计字典中的 'dec_time' 值
        stat_dict['all_dec_time'] = round(stat_dict['all_dec_time'], 3)  # 更新统计字典中的 'all_dec_time' 值
        stat_dict['base_dec_time'] = round(stat_dict['base_dec_time'], 3)  # 更新统计字典中的 'base_dec_time' 值
        return pc_rec, stat_dict  # 返回解压缩后的点云和统计字典



      def postprocess(self, pc_rec):
        """
        在点云被解压缩后进行后处理。
        """
        # 裁剪溢出值
        pc_rec = pc_rec.round().long()  # 将点云的坐标四舍五入并转换为长整型
        pc_rec[pc_rec[:, 0] >= self.res, 0] = self.res - 1  # 如果点云的x坐标大于等于分辨率，将其设置为分辨率减1
        pc_rec[pc_rec[:, 1] >= self.res, 1] = self.res - 1  # 如果点云的y坐标大于等于分辨率，将其设置为分辨率减1
        pc_rec[pc_rec[:, 2] >= self.res, 2] = self.res - 1  # 如果点云的z坐标大于等于分辨率，将其设置为分辨率减1
        pc_rec[pc_rec[:, 0] < 0, 0] = 0  # 如果点云的x坐标小于0，将其设置为0
        pc_rec[pc_rec[:, 1] < 0, 1] = 0  # 如果点云的y坐标小于0，将其设置为0
        pc_rec[pc_rec[:, 2] < 0, 2] = 0  # 如果点云的z坐标小于0，将其设置为0
    
        # 仅保留唯一的点
        pc_rec = pc_rec[:,0] * (self.res ** 2) + pc_rec[:, 1] * self.res + pc_rec[:, 2]  # 计算点云的一维坐标
        pc_rec = torch.unique(pc_rec)  # 保留唯一的点
        out0 = torch.floor(pc_rec / (self.res ** 2)).long()  # 计算点云的x坐标
        pc_rec = pc_rec - out0 * (self.res ** 2)  # 更新点云的一维坐标
        out1 = torch.floor(pc_rec / self.res).long()  # 计算点云的y坐标
        pc_rec = pc_rec - out1 * self.res  # 更新点云的一维坐标
        pc_rec = torch.cat([out0.unsqueeze(1), out1.unsqueeze(1), pc_rec.unsqueeze(1)], dim=1)  # 将点云的x坐标、y坐标和z坐标合并
        pc_rec = (pc_rec / self.scale - torch.tensor(self.translate, device=pc_rec.device)).long()  # 对点云的坐标进行反归一化
        return pc_rec  # 返回处理后的点云

