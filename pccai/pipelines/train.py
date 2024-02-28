# 版权所有 (c) 2010-2022，InterDigital
# 保留所有权利。
# 请在根文件夹下查看LICENSE。

# 训练函数。
# aux优化器是为了与CompressAI的兼容性（如果使用的话）

import shutil  # 导入shutil库
import time  # 导入time库
import os  # 导入os库
import numpy as np  # 导入NumPy库
import yaml  # 导入yaml库

import torch  # 导入torch库
from torch.utils.tensorboard import SummaryWriter  # 从torch.utils.tensorboard导入SummaryWriter
from collections import deque  # 从collections导入deque

import torch.distributed as dist  # 导入torch.distributed作为dist
from torch.nn.parallel import DistributedDataParallel as DDP  # 从torch.nn.parallel导入DistributedDataParallel作为DDP

# 从PccAI加载不同的实用程序
from pccai.models import PccModelWithLoss  # 从pccai.models导入PccModelWithLoss
from pccai.optim.utils import configure_optimization  # 从pccai.optim.utils导入configure_optimization
from pccai.utils.syntax import SyntaxGenerator  # 从pccai.utils.syntax导入SyntaxGenerator
from pccai.dataloaders.point_cloud_dataset import point_cloud_dataloader  # 从pccai.dataloaders.point_cloud_dataset导入point_cloud_dataloader
from pccai.utils.misc import load_state_dict_with_fallback  # 从pccai.utils.misc导入load_state_dict_with_fallback
import pccai.utils.logger as logger  # 导入pccai.utils.logger作为logger


def save_checkpoint(pccnet, optimizer, scheduler, epoch_state, aux_optimizer, aux_scheduler, opt, checkpoint_name):
    if opt.ddp and dist.get_rank() != 0: return # 在DDP模式下，只有当等级为0时才保存检查点
    data = {
        'net_state_dict': pccnet.module.state_dict(),
        'net_config': opt.net_config,
        'epoch_state': epoch_state,
    }
    if optimizer is not None:
        data['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        data['scheduler_state_dict'] = scheduler.state_dict()
    if aux_optimizer is not None:
        data['aux_optimizer_state_dict'] = aux_optimizer.state_dict()
    if aux_scheduler is not None:
        data['aux_scheduler_state_dict'] = aux_scheduler.state_dict()
    if (aux_optimizer is not None) or (aux_scheduler is not None):
        pccnet.module.pcc_model.update()
    torch.save(data, checkpoint_name)  # 保存数据到检查点


def load_checkpoint(checkpoint_path, with_optim, with_epoch_state, pccnet, epoch_state, optimizer=None,
                    scheduler=None, aux_optimizer=None, aux_scheduler=None):
    checkpoint = torch.load(checkpoint_path)  # 加载检查点
    
    if with_epoch_state:  # 如果with_epoch_state为真
        epoch_state.update(checkpoint.get('epoch_state', {}))  # 更新epoch_state

    load_state_dict_with_fallback(pccnet.module, checkpoint['net_state_dict'])  # 加载状态字典
    
    logger.log.info("Existing model %s loaded.\n" % (checkpoint_path))  # 记录信息
    
    if with_optim: # 如果需要，加载优化器和调度器
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if aux_optimizer is not None:
            aux_optimizer.load_state_dict(checkpoint['aux_optimizer_state_dict'])
        if aux_scheduler is not None:
            aux_scheduler.load_state_dict(checkpoint['aux_scheduler_state_dict'])
        logger.log.info("Optimization parameters loaded.\n")  # 记录信息

# 这段代码主要用于训练函数。其中，save_checkpoint函数用于保存检查点，load_checkpoint函数用于加载检查点。

def train_one_epoch(pccnet, dataloader, optimizer, aux_optimizer, writer, batch_total, opt):
    """使用模型、指定的损失、优化器、调度器等训练一个时期。"""

    pccnet.train() # 将模型设置为训练模式
    avg_loss = {}  # 初始化平均损失为字典
    len_data = len(dataloader)  # 获取数据加载器的长度
    batch_id = None  # 初始化批次id为None

    # 迭代训练过程
    for batch_id, points in enumerate(dataloader):
        if (points.shape[0] < dataloader.batch_size):  # 如果点的形状小于批次大小
            batch_id -= 1  # 批次id减1
            break  # 跳出循环

        points = points.cuda()  # 将点转移到GPU
        optimizer.zero_grad()  # 将优化器的梯度归零
        if aux_optimizer is not None:  # 如果aux_optimizer不为空
            aux_optimizer.zero_grad()  # 将aux_optimizer的梯度归零

        # 对主损失进行前向和后向传播
        output = pccnet(points)  # 获取输出

        loss = output['loss']  # 获取损失
        for k, v in loss.items(): loss[k] = torch.mean(v)  # 对损失进行平均

        loss['loss'].backward() # 计算梯度
        if opt.optim_config['clip_max_norm'] > 0:  # 如果剪裁最大范数大于0
            torch.nn.utils.clip_grad_norm_(pccnet.parameters(), opt.optim_config['clip_max_norm'])  # 对梯度进行剪裁
        optimizer.step() # 更新参数

        # 对辅助损失进行前向和后向传播
        if aux_optimizer is not None:  # 如果aux_optimizer不为空
            aux_loss = pccnet.module.pcc_model.aux_loss()  # 获取辅助损失
            aux_loss.backward()  # 计算辅助损失的梯度
            aux_optimizer.step()  # 更新参数

        # 记录结果
        for k in loss:
            avg_loss[k] = avg_loss.get(k, 0) + loss[k].item()  # 更新平均损失

        if batch_id % opt.print_freq == 0:  # 如果批次id能被打印频率整除
            message = '    batch count: %d/%d, iter: %d, ' % (batch_id, len_data, batch_id)  # 创建消息
            for k, v in loss.items(): 
                message += '%s: %f, ' % (k, v)  # 更新消息
                if writer is not None: writer.add_scalar('batch/' + k, v, batch_total)  # 如果writer不为空，添加标量
            if aux_optimizer is not None:  # 如果aux_optimizer不为空
                message += 'aux_loss: %f, ' % (aux_loss.item())  # 更新消息
            logger.log.info(message[:-2])  # 记录信息

        # 写下点云，只支持同质批处理
        if opt.pc_write_freq > 0 and batch_total % opt.pc_write_freq == 0 and opt.hetero == False:  # 如果点云写频率大于0且批次总数能被点云写频率整除且opt.hetero为False
            labels = np.concatenate((np.ones(output['x_hat'].shape[1]), np.zeros(points.shape[1])), axis=0).tolist()  # 创建标签
            if writer is not None: writer.add_embedding(torch.cat((output['x_hat'][0, 0:, 0:3], points[0, :, 0:3]), 0), 
                global_step=batch_total, metadata=labels, tag="point_cloud")  # 如果writer不为空，添加嵌入

        # torch.cuda.empty_cache() # 清空缓存
        batch_total += 1  # 批次总数加1

    for k in avg_loss.keys(): avg_loss[k] = avg_loss[k] / (batch_id + 1) # 计算平均损失
    return avg_loss, batch_total
    # 这段代码主要用于训练函数。其中，train_one_epoch函数用于训练一个时期，pccnet.train()用于将模型设置为训练模式，optimizer.zero_grad()和aux_optimizer.zero_grad()用于将优化器的梯度归零，
    # output = pccnet(points)用于获取输出，loss['loss'].backward()用于计算梯度，optimizer.step()和aux_optimizer.step()用于更新参数，avg_loss[k] = avg_loss.get(k, 0) + loss[k].item()用于更新平均损失，
    # batch_total += 1用于增加批次总数，最后返回平均损失和批次总数。


def validate_one_epoch(pccnet, dataloader, print_freq):
    """使用模型和指定的损失等验证一个时期。"""

    pccnet.eval() # 将模型设置为评估模式
    avg_loss = {}  # 初始化平均损失为字典
    len_data = len(dataloader)  # 获取数据加载器的长度
    batch_id = None  # 初始化批次id为None

    # 迭代验证过程
    for batch_id, points in enumerate(dataloader):
        
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
            message = '    batch count: %d/%d, iter: %d, ' % (batch_id, len_data, batch_id)  # 创建消息
            for k, v in loss.items(): message += '%s: %f, ' % (k, v)  # 更新消息
            logger.log.info(message[:-2])  # 记录信息

    for k in avg_loss.keys(): avg_loss[k] = avg_loss[k] / (batch_id + 1) # 计算平均损失
    return avg_loss  # 返回平均损失


def train_pccnet(opt):
    """训练一个点云压缩网络。"""
    
    opt.phase='train'  # 设置阶段为训练
    syntax = SyntaxGenerator(opt)  # 创建语法生成器
    pccnet = PccModelWithLoss(opt.net_config, syntax, opt.optim_config['loss_args'])  # 创建PccModelWithLoss
    logger.log.info("%d GPU(s) will be used for training." % opt.device_count)  # 记录信息
    if opt.ddp:  # 如果opt.ddp为真
        # 使用DDP包装自动编码器
        pccnet.to(opt.device)  # 将pccnet转移到设备
        pccnet = DDP(pccnet, device_ids=[opt.device], output_device=[opt.device], find_unused_parameters=True)  # 创建DDP
    else:  # 否则
        # 使用DP包装自动编码器
        pccnet = torch.nn.DataParallel(pccnet)  # 创建DataParallel
        pccnet.to(torch.device("cuda:" + str(opt.device))) # 0是主设备

    # 处理数据集
    _, train_dataloader = point_cloud_dataloader(opt.train_data_config, syntax, opt.ddp)  # 获取训练数据加载器
    if opt.val_freq > 0:  # 如果验证频率大于0
        _, val_dataloader = point_cloud_dataloader(opt.val_data_config, syntax, opt.ddp)  # 获取验证数据加载器

    # 这段代码主要用于训练函数。其中，validate_one_epoch函数用于验证一个时期，pccnet.eval()用于将模型设置为评估模式，output = pccnet(points)用于获取输出，loss['loss'].backward()用于计算梯度，
    # optimizer.step()和aux_optimizer.step()用于更新参数，avg_loss[k] = avg_loss.get(k, 0) + loss[k].item()用于更新平均损失，batch_total += 1用于增加批次总数，最后返回平均损失和批次总数。
    
    
 # 配置优化相关的内容
    optimizer, scheduler, aux_optimizer, aux_scheduler = configure_optimization(pccnet, opt.optim_config)  # 配置优化
    epoch_state = { 'last_epoch': -1, 'total_epoch': -1 }  # 初始化epoch_state
    # 如果给定了一个已保存的模型，则加载它
    if opt.checkpoint != '':
        load_checkpoint(checkpoint_path=opt.checkpoint, with_optim=opt.checkpoint_optim_config, with_epoch_state=opt.checkpoint_epoch_state,
                        pccnet=pccnet, epoch_state=epoch_state, 
                        optimizer=optimizer, scheduler=scheduler,
                        aux_optimizer=aux_optimizer, aux_scheduler=aux_scheduler)  # 加载检查点
    
        # 修复指定模块的权重
        modules = {module_name: param for module_name, param in pccnet.module.named_modules()}  # 获取模块
        params = {param_name: param for param_name, param in pccnet.module.named_parameters()}  # 获取参数
        for fix_module in opt.fix_modules:  # 对于每一个需要修复的模块
            logger.log.info('Fix the weights of %s' % fix_module)  # 记录信息
            param = modules.get(fix_module, params.get(fix_module, None))  # 获取参数
            param.requires_grad_(False)  # 设置参数的requires_grad为False

    # 创建一个tensorboard writer
    writer = SummaryWriter(comment='_' + opt.exp_name) \
        if opt.tf_summary and not(opt.ddp and dist.get_rank() != 0) else None  # 创建SummaryWriter

    # 开始训练过程
    batch_total = 0  # 初始化批次总数为0
    checkpoint_queue = deque()  # 创建一个双端队列
    mdl_cnt = sum(p.numel() for p in pccnet.parameters() if p.requires_grad)  # 计算模型参数的数量
    logger.log.info('Model parameter count %d.' % mdl_cnt)  # 记录信息

    t = time.monotonic()  # 获取当前时间
    epoch = epoch_state['last_epoch']  # 获取最后一个时期
    total_epoch = epoch_state['total_epoch']  # 获取总时期
    while epoch < opt.optim_config['n_epoch']:  # 当时期小于n_epoch时
        epoch_state['last_epoch'] += 1  # 最后一个时期加1
        epoch_state['total_epoch'] += 1  # 总时期加1
        epoch = epoch_state['last_epoch']  # 更新时期
        total_epoch = epoch_state['total_epoch']  # 更新总时期
        if opt.ddp: train_dataloader.sampler.set_epoch(epoch) # 这是为了娱乐DDP

        # 执行一个时期的训练
        lr = optimizer.param_groups[0]['lr']  # 获取学习率
        aux_lr = aux_optimizer.param_groups[0]['lr'] if aux_optimizer is not None else None  # 获取辅助学习率
        logger.log.info(f'Training at epoch {epoch} (total {total_epoch}) with lr {lr}' +
                        (f' and aux_lr {aux_lr}' if aux_scheduler is not None else ''))  # 记录信息
        avg_loss, batch_total = train_one_epoch(pccnet, train_dataloader, optimizer, aux_optimizer, writer, batch_total, opt)  # 训练一个时期

        if scheduler is not None:  # 如果调度器不为空
            scheduler.step()  # 调度器步进
        if aux_scheduler is not None:  # 如果辅助调度器不为空
            aux_scheduler.step()  # 辅助调度器步进
        elapse = time.monotonic() - t  # 计算经过的时间
        # 这段代码主要用于训练函数。其中，validate_one_epoch函数用于验证一个时期，pccnet.eval()用于将模型设置为评估模式，output = pccnet(points)用于获取输出，loss['loss'].backward()用于计算梯度，optimizer.step()和aux_optimizer.step()用于更新参数，
        # avg_loss[k] = avg_loss.get(k, 0) + loss[k].item()用于更新平均损失，batch_total += 1用于增加批次总数，最后返回平均损失和批次总数。

        
# 记录训练结果
        message = 'Epoch: %d/%d --- time: %f, lr: %f, ' % (epoch, total_epoch, elapse, lr)  # 创建消息
        for k, v in avg_loss.items():  # 对于平均损失中的每一项
            message += 'avg_%s: %f, ' % (k, v)  # 更新消息
            if writer is not None: writer.add_scalar('epoch/avg_' + k, v, epoch)  # 如果writer不为空，添加标量
        logger.log.info(message[:-2] + '\n')  # 记录信息
        if writer is not None: writer.add_scalar('epoch/learning_rate', lr, epoch)  # 如果writer不为空，添加标量

        # 如果需要，保存新的检查点
        if epoch % opt.save_checkpoint_freq == 0 or epoch == opt.optim_config['n_epoch'] - 1:  # 如果时期能被保存检查点频率整除或时期等于n_epoch减1
            checkpoint_name = os.path.join(opt.exp_folder, opt.save_checkpoint_prefix + str(epoch) + '.pth')  # 创建检查点名称
            save_checkpoint(pccnet, optimizer, scheduler, epoch_state, aux_optimizer, aux_scheduler, opt, checkpoint_name)  # 保存检查点

            if not(opt.ddp and dist.get_rank() != 0):  # 如果opt.ddp为假或等级不等于0
                shutil.copyfile(checkpoint_name, os.path.join(opt.exp_folder, opt.save_checkpoint_prefix + 'newest.pth'))  # 复制文件
            logger.log.info('Current checkpoint saved to %s.\n' % (checkpoint_name))  # 记录信息

            # 维护总检查点计数
            checkpoint_queue.append((epoch, checkpoint_name))  # 将元组添加到检查点队列
            if len(checkpoint_queue) > opt.save_checkpoint_max:  # 如果检查点队列的长度大于保存检查点最大值
                _, pop_checkpoint_name = checkpoint_queue.popleft()  # 弹出检查点队列的左端元素
                if os.path.exists(pop_checkpoint_name): os.remove(pop_checkpoint_name)  # 如果弹出的检查点名称存在，删除它

        # 执行一个时期的验证并记录结果
        if opt.val_freq > 0 and epoch % opt.val_freq == 0:  # 如果验证频率大于0且时期能被验证频率整除
            logger.log.info('Validation at epoch %d' % epoch)  # 记录信息
            avg_loss_val = validate_one_epoch(pccnet, val_dataloader, opt.val_print_freq)  # 验证一个时期

            # 记录验证结果
            message = 'Validation --- '  # 创建消息
            for k, v in avg_loss_val.items():  # 对于平均验证损失中的每一项
                message += 'avg_val_%s: %f, ' % (k, v)  # 更新消息
                if writer is not None: writer.add_scalar('epoch/avg_val_' + k, v, epoch)  # 如果writer不为空，添加标量
            logger.log.info(message[:-2] + '\n')  # 记录信息

    if writer is not None: writer.close()  # 如果writer不为空，关闭writer
    return avg_loss  # 返回平均损失

# 这段代码主要用于训练函数。其中，validate_one_epoch函数用于验证一个时期，pccnet.eval()用于将模型设置为评估模式，output = pccnet(points)用于获取输出，loss['loss'].backward()用于计算梯度，optimizer.step()和aux_optimizer.step()用于更新参数，
# avg_loss[k] = avg_loss.get(k, 0) + loss[k].item()用于更新平均损失，batch_total += 1用于增加批次总数，最后返回平均损失和批次总数。

def load_train_config(opt):
    """加载训练的所有配置文件。"""

    # 加载训练和验证数据配置
    with open(opt.train_data_config[0], 'r') as file:
        train_data_config = yaml.load(file, Loader=yaml.FullLoader)
    opt.train_data_config[0] = train_data_config
    if opt.val_data_config != '':
        with open(opt.val_data_config[0], 'r') as file:
            val_data_config = yaml.load(file, Loader=yaml.FullLoader)
        opt.val_data_config[0] = val_data_config

    # 加载优化配置
    with open(opt.optim_config, 'r') as file:
        optim_config = yaml.load(file, Loader = yaml.FullLoader)

    # R-D权重和学习率是特殊参数，可以从输入参数中覆盖
    if opt.alpha is not None:
        optim_config['loss_args']['alpha'] = opt.alpha
    else:
        logger.log.info('alpha from optim config: ' + str(optim_config['loss_args']['alpha']))
    if opt.beta is not None:
        optim_config['loss_args']['beta'] = opt.beta
    else:
        logger.log.info('beta from optim config: ' + str(optim_config['loss_args']['beta']))
    if opt.lr is not None:
        optim_config['main_args']['lr'] = opt.lr
    else:
        logger.log.info('lr from optim config: ' + str(optim_config['main_args']['lr']))

    if 'aux_args' not in optim_config.keys(): # 如果'aux_args'未指定，则使用'main_args'
        optim_config['aux_args'] = optim_config['main_args']
    if opt.lr_aux is not None: optim_config['aux_args']['lr'] = opt.lr_aux
    opt.optim_config = optim_config

    # 加载网络配置
    with open(opt.net_config, 'r') as file:
        net_config = yaml.load(file, Loader=yaml.FullLoader)

    # 方法特定参数
    net_config['modules']['scaling_ratio'] = opt.scaling_ratio
    net_config['modules']['point_mul'] = opt.point_mul
    net_config['modules']['skip_mode'] = opt.skip_mode

    opt.net_config = net_config

    return opt


if __name__ == "__main__":
    logger.log.error('Not implemented.')

# 这段代码主要用于加载训练的所有配置文件。其中，load_train_config函数用于加载训练的所有配置文件，with open(opt.train_data_config[0], 'r') as file:用于打开训练数据配置文件，train_data_config = yaml.load(file, Loader=yaml.FullLoader)用于加载训练数据配置，
# opt.train_data_config[0] = train_data_config用于更新训练数据配置，if opt.val_data_config != '':用于判断是否存在验证数据配置，with open(opt.val_data_config[0], 'r') as file:用于打开验证数据配置文件，val_data_config = yaml.load(file, Loader=yaml.FullLoader)用于加载验证数据配置，
# opt.val_data_config[0] = val_data_config用于更新验证数据配置，with open(opt.optim_config, 'r') as file:用于打开优化配置文件，optim_config = yaml.load(file, Loader = yaml.FullLoader)用于加载优化配置，if opt.alpha is not None:用于判断是否存在alpha参数，optim_config['loss_args']['alpha'] = opt.alpha用于更新alpha参数，
# else:用于处理不存在alpha参数的情况，logger.log.info('alpha from optim config: ' + str(optim_config['loss_args']['alpha']))用于记录信息，if opt.beta is not None:用于判断是否存在beta参数，optim_config['loss_args']['beta'] = opt.beta用于更新beta参数，else:用于处理不存在beta参数的情况，
# logger.log.info('beta from optim config: ' + str(optim_config['loss_args']['beta']))用于记录信息，if opt.lr is not None:用于判断是否存在lr参数，optim_config['main_args']['lr'] = opt.lr用于更新lr参数，else:用于处理不存在lr参数的情况，logger.log.info('lr from optim config: ' + str(optim_config['main_args']['lr']))用于记录信息，
# if 'aux_args' not in optim_config.keys():用于判断是否存在aux_args参数，optim_config['aux_args'] = optim_config['main_args']用于更新aux_args参数，if opt.lr_aux is not None:用于判断是否存在lr_aux参数，optim_config['aux_args']['lr'] = opt.lr_aux用于更新lr_aux参数，opt.optim_config = optim_config用于更新优化配置，
# with open(opt.net_config, 'r') as file:用于打开网络配置文件，net_config = yaml.load(file, Loader=yaml.FullLoader)用于加载网络配置，net_config['modules']['scaling_ratio'] = opt.scaling_ratio用于更新scaling_ratio参数，net_config['modules']['point_mul'] = opt.point_mul用于更新point_mul参数，
# net_config['modules']['skip_mode'] = opt.skip_mode用于更新skip_mode参数，opt.net_config = net_config用于更新网络配置，return opt用于返回opt，if __name__ == "__main__":用于判断是否为主程序，logger.log.error('Not implemented.')用于记录错误信息。
