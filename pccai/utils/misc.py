import numpy as np  # 导入NumPy库
import pccai.utils.logger as logger  # 导入pccai.utils.logger作为logger
from plyfile import PlyData, PlyElement  # 从plyfile导入PlyData和PlyElement

def pc_write(pc, file_name):  # 定义写点云的函数
    pc_np = pc.T.cpu().numpy()  # 将点云转换为NumPy数组
    vertex = list(zip(pc_np[0], pc_np[1], pc_np[2]))  # 创建顶点列表
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])  # 将顶点列表转换为NumPy数组
    elements = PlyElement.describe(vertex, "vertex")  # 描述元素
    PlyData([elements]).write(file_name)  # 写入文件
    return  # 返回

def pc_read(filename):  # 定义读点云的函数
    ply_raw = PlyData.read(filename)['vertex'].data  # 读取文件
    pc = np.vstack((ply_raw['x'], ply_raw['y'], ply_raw['z'])).transpose()  # 创建点云
    return np.ascontiguousarray(pc)  # 返回点云

def pt_to_np(tensor):  # 定义将PyTorch张量转换为NumPy数组的函数
    """将PyTorch张量转换为NumPy数组。"""

    return tensor.contiguous().cpu().detach().numpy()  # 返回NumPy数组

def load_state_dict_with_fallback(obj, dict):  # 定义加载检查点的函数
    """使用回退加载检查点。"""

    try:  # 尝试
        obj.load_state_dict(dict)  # 加载检查点
    except RuntimeError as e:  # 如果出现运行时错误
        logger.log.exception(e)  # 记录异常
        logger.log.info(f'Strict load_state_dict has failed. Attempting in non strict mode.')  # 记录信息
        obj.load_state_dict(dict, strict=False)  # 以非严格模式加载检查点

#这段代码主要用于处理点云数据，包括读取点云数据、写入点云数据、将PyTorch张量转换为NumPy数组以及加载检查点。
