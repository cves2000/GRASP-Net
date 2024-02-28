# 版权所有 (c) 2010-2022, InterDigital
# 保留所有权利。

# 请参阅根文件夹下的LICENSE。

# 训练一个点云压缩模型

import random
import os
import torch
import sys
import socket
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')  # 将当前文件的父目录添加到系统路径

# 多进程工具
import torch.multiprocessing as mp
import torch.distributed as dist

# 从PccAI加载不同的工具
from pccai.utils.option_handler import TrainOptionHandler
import pccai.utils.logger as logger
from pccai.pipelines.train import *


def setup(rank, world_size, master_address, master_port):
    """如果需要，设置DDP进程，每个进程将被分配到一个GPU。"""

    # 首先寻找一个可用的端口
    tmp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        loc = (master_address, master_port)
        res = tmp_socket.connect_ex(loc)
        if res != 0: break # 找到一个端口
        else: master_port += 1

    # 初始化进程组
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['MASTER_ADDR'] = master_address
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """销毁所有进程。"""

    dist.destroy_process_group()


def train_main(device, opt):
    """主训练包装器。"""

    # 初始化一个全局日志记录器，然后打印出所有的选项
    logger.create_logger(opt.exp_folder, opt.log_file, opt.log_file_only)
    option_handler = TrainOptionHandler()
    option_handler.print_options(opt)
    opt = load_train_config(opt)
    opt.device = device
    opt.device_count = torch.cuda.device_count()
    if opt.ddp: setup(device, opt.device_count, opt.master_address, opt.master_port)

    # 进行实际的训练
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)
    avg_loss = train_pccnet(opt)
    logger.log.info('训练会话 %s 已完成。\n' % opt.exp_name)
    logger.destroy_logger()
    if opt.ddp: cleanup()


if __name__ == "__main__":

    # 解析选项并进行训练
    option_handler = TrainOptionHandler()
    opt = option_handler.parse_options()

    # 创建一个文件夹来保存模型和日志
    if not os.path.exists(opt.exp_folder):
        os.makedirs(opt.exp_folder)
    if opt.ddp:
        mp.spawn(train_main, args=(opt,), nprocs=torch.cuda.device_count(), join=True)
    else:
        train_main(0, opt)
# 这段代码主要用于训练一个点云压缩模型。其中，setup函数用于设置分布式数据并行（DDP）进程，
# cleanup函数用于销毁所有进程，train_main函数是主训练包装器，它初始化一个全局日志记录器，打印出所有的选项，加载训练配置，设置设备，并进行实际的训练。
