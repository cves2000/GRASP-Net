# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Test the point cloud compression model, this is to verify the loss on the datasets but not for actual compression

import time
import os
import yaml
import torch
import numpy as np

# Load different utilities from PccAI
from pccai.models import PccModelWithLoss
from pccai.dataloaders.point_cloud_dataset import point_cloud_dataloader
from pccai.utils.syntax import SyntaxGenerator
from pccai.utils.misc import pc_write, load_state_dict_with_fallback
import pccai.utils.logger as logger

def test_one_epoch(pccnet, dataset, dataloader, syntax, gen_bitstream, print_freq, pc_write_freq, pc_write_prefix, exp_folder=None):
    """使用给定的模型、指定的损失函数和给定的数据集等测试一个时期。"""

    # 执行一个时期的测试
    avg_loss = {}  # 初始化平均损失为字典
    avg_real = {'xyz_loss': 0, 'bpp_loss': 0, 'loss': 0} if gen_bitstream else None  # 如果gen_bitstream为真，初始化avg_real为字典，否则为None
    len_data = len(dataloader)  # 获取数据加载器的长度
    batch_id = None  # 初始化批次id为None
    if syntax.hetero:  # 如果syntax.hetero为真
        syntax_gt = syntax.syntax_gt  # 获取syntax_gt
        syntax_rec = syntax.syntax_rec  # 获取syntax_rec

    for batch_id, points in enumerate(dataloader):  # 对于数据加载器中的每一批次
        
        if(points.shape[0] < dataloader.batch_size):  # 如果点的形状小于批次大小
            batch_id -= 1  # 批次id减1
            break  # 跳出循环
        points = points.cuda()  # 将点转移到GPU

        # 推理和计算损失
        with torch.no_grad():  # 不计算梯度
            output = pccnet(points)  # 获取输出
            loss = output['loss']  # 获取损失
            for k, v in loss.items(): loss[k] = torch.mean(v)  # 对损失进行平均

        # 记录结果
        if batch_id == 0:  # 如果批次id为0
            for k, v in loss.items(): avg_loss[k] = v.item()  # 更新平均损失
        else:  # 否则
            for k, v in loss.items(): avg_loss[k] += v.item()  # 更新平均损失
        if batch_id % print_freq == 0:  # 如果批次id能被打印频率整除
            message = '    batch count: %d/%d, ' % (batch_id, len_data)  # 创建消息
            for k, v in loss.items(): message += '%s: %f, ' % (k, v)  # 更新消息
            logger.log.info(message[:-2])  # 记录信息

        # 执行REAL压缩，这部分在异构模式下有用
        if gen_bitstream:  # 如果gen_bitstream为真
            # 压缩然后解压缩
            with torch.no_grad():  # 不计算梯度
                cmp_out, meta_data = pccnet.pcc_model.compress(points) # 压缩
                rec_real, meta_data = pccnet.pcc_model.decompress(cmp_out['strings'], cmp_out['shape'], meta_data) # 解压缩
                if syntax.hetero:  # 如果syntax.hetero为真
                    rec_real = torch.hstack([rec_real, meta_data])  # 水平堆叠rec_real和meta_data
                elif len(rec_real.shape) == 2:  # 如果rec_real的形状等于2
                        rec_real = rec_real.unsqueeze(0)  # 在rec_real的第0维增加一个维度
    # 代码主要用于测试一个时期。其中，test_one_epoch函数用于测试一个时期，pccnet.eval()用于将模型设置为评估模式，output = pccnet(points)用于获取输出，
    # loss['loss'].backward()用于计算梯度，optimizer.step()和aux_optimizer.step()用于更新参数，
    # avg_loss[k] = avg_loss.get(k, 0) + loss[k].item()用于更新平均损失，batch_total += 1用于增加批次总数，最后返回平均损失和批次总数。
        
           # 计算损失并记录结果
            bpp_loss_batch = 0 # 当前批次的每点位数
            for i in range(len(cmp_out['strings'][0])):
                bpp_loss_batch += len(cmp_out['strings'][0][i]) * 8
            if syntax.hetero:
                bpp_loss_batch /= torch.sum(points[:, :, syntax_gt['block_pntcnt']] > 0)
            else:
                bpp_loss_batch /= dataloader.batch_size * dataset.num_points
            xyz_loss_batch = {}
            pccnet.loss.xyz_loss(xyz_loss_batch, points, output) # 当前批次的失真损失
            xyz_loss_batch = xyz_loss_batch['xyz_loss'].item()
            real_loss_batch = pccnet.loss.alpha * xyz_loss_batch + pccnet.loss.beta * bpp_loss_batch
            avg_real['bpp_loss'] += bpp_loss_batch
            avg_real['xyz_loss'] += xyz_loss_batch
            avg_real['loss'] += real_loss_batch
            if batch_id % print_freq == 0:
                logger.log.info('        real stat. ---- bpp_loss: %f, xyz_loss: %f, loss: %f' % (bpp_loss_batch, xyz_loss_batch, real_loss_batch))
            
            # 如果需要，写下点云
            if pc_write_freq > 0 and batch_id % pc_write_freq == 0: # 如果需要，写点云
                filename_rec_real = os.path.join(exp_folder, pc_write_prefix + str(batch_id) + "_rec_real.ply")
                if syntax.hetero:
                    pc_rec_real = rec_real[torch.cumsum(rec_real[:, syntax_rec['pc_start']], dim=0) == 1, 
                        syntax_rec['xyz'][0] : syntax_rec['xyz'][1] + 1]
                    pc_write(pc_rec_real, filename_rec_real)
                else:
                    pc_write(rec_real[0], filename_rec_real)

    # 记录结果
    for k in avg_loss.keys(): avg_loss[k] = avg_loss[k] / (batch_id + 1) # 平均损失

    # 如果执行了REAL压缩，记录结果
    if gen_bitstream:
        for k in avg_real.keys(): avg_real[k] = avg_real[k] / (batch_id + 1) # 平均损失

    return avg_loss, avg_real
# 这段代码主要用于测试一个时期。其中，test_one_epoch函数用于测试一个时期，pccnet.eval()用于将模型设置为评估模式，output = pccnet(points)用于获取输出，
# loss['loss'].backward()用于计算梯度，optimizer.step()和aux_optimizer.step()用于更新参数，avg_loss[k] = avg_loss.get(k, 0) + loss[k].item()用于更新平均损失，
# batch_total += 1用于增加批次总数，最后返回平均损失和批次总数。

def test_pccnet(opt):
    """测试一个点云压缩网络。这不是为了实际的点云压缩，而是为了测试训练过的网络。"""

    logger.log.info("%d GPU(s) 将被用于测试." % torch.cuda.device_count())  # 记录信息，将用于测试的GPU数量
    opt.phase = 'test'  # 设置阶段为测试

    # 加载一个现有的检查点
    checkpoint = torch.load(opt.checkpoint)  # 加载检查点
    if opt.checkpoint_net_config == True:  # 如果检查点网络配置为真
        opt.net_config = checkpoint['net_config']  # 从检查点加载网络配置
        logger.log.info("从检查点 %s 加载模型配置." % opt.checkpoint)  # 记录信息，从检查点加载模型配置
        logger.log.info(opt.net_config)  # 记录信息，网络配置
    syntax = SyntaxGenerator(opt)  # 生成语法

    pccnet = PccModelWithLoss(opt.net_config, syntax, opt.optim_config['loss_args'])  # 创建PccModelWithLoss对象
    state_dict = checkpoint['net_state_dict']  # 从检查点获取状态字典
    for _ in range(len(state_dict)):  # 对于状态字典中的每一项
        k, v = state_dict.popitem(False)  # 弹出一项
        state_dict[k[len('.pcc_model'):]] = v  # 更新状态字典
    load_state_dict_with_fallback(pccnet.pcc_model, state_dict)  # 加载状态字典
    logger.log.info("从检查点 %s 加载模型权重.\n" % opt.checkpoint)  # 记录信息，从检查点加载模型权重
    device = torch.device("cuda:0")  # 设置设备为cuda:0
    pccnet.to(device)  # 将pccnet转移到设备
    pccnet.eval() # 为了让噪声添加到码字中，不应将其设置为评估模式

    # 杂项配置
    test_dataset, test_dataloader = point_cloud_dataloader(opt.test_data_config, syntax) # 配置数据集

    # 开始测试过程
    t = time.monotonic()  # 获取当前时间
    avg_loss, avg_real = test_one_epoch(pccnet, test_dataset, test_dataloader, syntax,
        opt.gen_bitstream, opt.print_freq, opt.pc_write_freq, opt.pc_write_prefix, opt.exp_folder)  # 测试一个时期
    elapse = time.monotonic() - t  # 计算经过的时间

    # 记录测试结果
    message = 'Validation --- time: %f, ' % elapse  # 创建消息
    for k, v in avg_loss.items(): message += 'avg_%s: %f, ' % (k, v)  # 更新消息
    logger.log.info(message[:-2])  # 记录信息
    if opt.gen_bitstream:  # 如果gen_bitstream为真
        message = 'real stat --- '  # 创建消息
        for k, v in avg_real.items(): message += 'avg_%s: %f, ' % (k, v)  # 更新消息
        logger.log.info(message[:-2])  # 记录信息

    return avg_loss  # 返回平均损失


def load_test_config(opt):
    """加载测试的所有配置文件。"""

    # 加载测试数据配置
    with open(opt.test_data_config[0], 'r') as file:  # 打开测试数据配置文件
        test_data_config = yaml.load(file, Loader=yaml.FullLoader)  # 加载测试数据配置
    opt.test_data_config[0] = test_data_config  # 更新测试数据配置

    # 加载优化配置
    with open(opt.optim_config, 'r') as file:  # 打开优化配置文件
        optim_config = yaml.load(file, Loader = yaml.FullLoader)  # 加载优化配置
        if opt.alpha is not None:  # 如果alpha不为空
            optim_config['loss_args']['alpha'] = opt.alpha  # 更新alpha
        else:  # 否则
            logger.log.info('alpha from optim config: ' + str(optim_config['loss_args']['alpha']))  # 记录信息，从优化配置获取alpha
        if opt.beta is not None:  # 如果beta不为空
            optim_config['loss_args']['beta'] = opt.beta  # 更新beta
        else:  # 否则
            logger.log.info('beta from optim config: ' + str(optim_config['loss_args']['beta']))  # 记录信息，从优化配置获取beta
    opt.optim_config = optim_config  # 更新优化配置

    # 加载网络配置
    if opt.net_config != '':  # 如果网络配置不为空
        with open(opt.net_config, 'r') as file:  # 打开网络配置文件
            net_config = yaml.load(file, Loader=yaml.FullLoader)  # 加载网络配置
        opt.net_config = net_config  # 更新网络配置

    return opt  # 返回opt


if __name__ == "__main__":

    logger.log.error('Not implemented.')
