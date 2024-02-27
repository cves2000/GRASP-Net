# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Import all the modules to be used here
from pccai.models.modules.pointnet import PointNet, PointNetHetero
from pccai.models.modules.mlpdecoder import MlpDecoder, MlpDecoderHetero
from pccai.models.modules.pointnet_residual import PointResidualEncoder
from pccai.models.modules.mlpdecoder_sparse import MlpDecoderSparse
from pccai.models.modules.spcnn_down import SparseCnnDown1, SparseCnnDown2
from pccai.models.modules.spcnn_up import SparseCnnUp1, SparseCnnUp2


def get_module_class(module_name, hetero=False):
    """
    从模块名称中检索模块类。
    """

    # 在此字典中列出所有模块及其字符串名称
    module_dict = {
        'pointnet': [PointNet, PointNetHetero], # pointnet
        'mlpdecoder': [MlpDecoder, MlpDecoderHetero], # mlpdecoder
        # 以下模块用于GRASP-Net
        'point_res_enc': [PointResidualEncoder, None],
        'mlpdecoder_sparse': [MlpDecoderSparse, None],
        'spcnn_down': [SparseCnnDown1, None],
        'spcnn_up': [SparseCnnUp1, None],
        'spcnn_down2': [SparseCnnDown2, None],
        'spcnn_up2': [SparseCnnUp2, None],
    }

    # 根据模块名称获取模块
    module = module_dict.get(module_name.lower(), None)
    assert module is not None, f'module {module_name} was not found, valid modules are: {list(module_dict.keys())}'
    try:
        # 根据是否为异构模式获取模块
        module = module[hetero]
    except IndexError as e:
        raise Exception(f'module {module_name} is not implemented for hetero={hetero}')

    return module
    # 在这段代码中，首先定义了一个字典module_dict，其中的键是模块名称，值是模块类的列表。然后，根据输入的模块名称module_name从字典中获取对应的模块类。
    # 如果输入的模块名称在字典中不存在，那么会抛出一个断言错误。
    # 接着，根据输入的异构标志hetero从模块类列表中获取对应的模块类。如果输入的异构标志超出了模块类列表的索引范围，那么会抛出一个索引错误。最后，返回获取到的模块类。
