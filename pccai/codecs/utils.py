# 配置点云编解码器的实用工具，包括导入所有要使用的编解码器，列出所有的编解码器，以及根据编解码器的名称获取对应的编解码器类
# 版权所有 (c) 2010-2022，InterDigital
# 保留所有权利。

# 请参阅根文件夹下的LICENSE。

# 点云编解码器的实用工具

import torch  # 导入torch库，torch是一个开源的机器学习库，提供了广泛的机器学习算法
import numpy as np  # 导入numpy库，numpy是Python的一个科学计算的库，提供了矩阵运算的功能

# 导入所有要使用的编解码器
from pccai.codecs.grasp_codec import GeoResCompressionCodec  # 从pccai.codecs.grasp_codec模块导入GeoResCompressionCodec类

# 在以下字典中列出所有的编解码器
codec_classes = {
    'grasp_codec': GeoResCompressionCodec  # 'grasp_codec'对应的编解码器类为GeoResCompressionCodec
}

def get_codec_class(codec_name):  # 定义一个函数，根据编解码器的名称获取对应的编解码器类
    codec = codec_classes.get(codec_name.lower(), None)  # 从字典中获取对应的编解码器类，如果没有找到则返回None
    assert codec is not None, f'codec class "{codec_name}" not found, valid codec classes are: {list(codec_classes.keys())}'  # 如果没有找到对应的编解码器类，则抛出异常
    return codec  # 返回找到的编解码器类
