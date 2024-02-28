# 版权所有 (c) 2010-2022, InterDigital
# 保留所有权利。

# 请参阅根文件夹下的LICENSE。

# 测试训练过的点云压缩模型

import multiprocessing
multiprocessing.set_start_method('spawn', True)  # 设置多进程的启动方式为'spawn'

import random
import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')  # 将当前文件的父目录添加到系统路径

# 从PccAI加载不同的工具
from pccai.utils.option_handler import TestOptionHandler
import pccai.utils.logger as logger
from pccai.pipelines.test import *


if __name__ == "__main__":

    # 解析选项并进行训练
    option_handler = TestOptionHandler()
    opt = option_handler.parse_options()

    # 创建一个文件夹来保存模型和日志
    if not os.path.exists(opt.exp_folder):
        os.makedirs(opt.exp_folder)

    # 初始化一个全局日志记录器，然后打印出所有的选项
    logger.create_logger(opt.exp_folder, opt.log_file, opt.log_file_only)
    option_handler.print_options(opt)
    opt = load_test_config(opt)

    # 进行实际的训练
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)
    avg_loss = test_pccnet(opt)
    logger.log.info('测试会话 %s 已完成。\n' % opt.exp_name)
    logger.destroy_logger()
# 这段代码主要用于测试一个训练过的点云压缩模型。其中，multiprocessing.set_start_method('spawn', True)用于设置多进程的启动方式为’spawn’，
# option_handler = TestOptionHandler()用于创建一个TestOptionHandler对象，opt = option_handler.parse_options()用于解析选项，
# if not os.path.exists(opt.exp_folder): os.makedirs(opt.exp_folder)用于创建一个文件夹来保存模型和日志，
# logger.create_logger(opt.exp_folder, opt.log_file, opt.log_file_only)用于初始化一个全局日志记录器，option_handler.print_options(opt)用于打印出所有的选项，
# opt = load_test_config(opt)用于加载测试配置，if opt.seed is not None: torch.manual_seed(opt.seed) random.seed(opt.seed)用于设置随机种子，
# avg_loss = test_pccnet(opt)用于测试点云压缩网络，logger.log.info('测试会话 %s 已完成。\n' % opt.exp_name)用于记录信息，测试会话已完成，logger.destroy_logger()用于销毁日志记录器。
