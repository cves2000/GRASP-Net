#PCC编解码器的基础类，包括类的初始化方法、压缩方法和解压缩方法
# 版权所有 (c) 2010-2022，InterDigital
# 保留所有权利。

# 请参阅根文件夹下的LICENSE。

import numpy as np  # 导入numpy库，numpy是Python的一个科学计算的库，提供了矩阵运算的功能

class PccCodecBase:  # 定义一个名为PccCodecBase的类
    """A base class of PCC codec. User needs to implement the compress() and decompress() method."""

    def __init__(self, codec_config, pccnet, syntax):  # 类的初始化方法
        self.translate = codec_config['translate']  # 从编解码器配置中获取平移参数
        self.scale = codec_config['scale']  # 从编解码器配置中获取缩放参数
        self.hetero = syntax.hetero  # 获取语法的异质性参数
        self.phase = syntax.phase  # 获取语法的阶段参数
        self.pccnet = pccnet  # 获取pccnet参数

    def compress(self, points, tag):  # 定义压缩方法，需要用户实现
        """Compression method."""
        raise NotImplementedError()  # 如果用户没有实现这个方法，调用时会抛出NotImplementedError异常

    def decompress(self, file_name):  # 定义解压缩方法，需要用户实现
        """Decompression method."""
        raise NotImplementedError()  # 如果用户没有实现这个方法，调用时会抛出NotImplementedError异常

    
    
    def decompress(self, file_name):
        """Decompression method."""

        raise NotImplementedError()
