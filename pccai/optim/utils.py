# Copyright (c) 2010-2022, InterDigital
# 版权所有 (c) 2010-2022，InterDigital
# All rights reserved. 
# 保留所有权利。
# See LICENSE under the root folder.
# 请在根文件夹下查看LICENSE。

# Utilities related to network optimization
# 与网络优化相关的实用程序

import torch  # 导入PyTorch库
import torch.optim as optim  # 导入PyTorch优化库

# Import all the loss classes to be used
# 导入所有要使用的损失类
from pccai.optim.cd_sparse import ChamferDistSparse  # 从pccai.optim.cd_sparse导入ChamferDistSparse

# List the all the loss classes in the following dictionary 
# 在以下字典中列出所有的损失类
loss_classes = {
    'cd_sparse': ChamferDistSparse
}

def get_loss_class(loss_name):  # 定义获取损失类的函数
    loss = loss_classes.get(loss_name.lower(), None)  # 从损失类字典中获取指定的损失类
    assert loss is not None, f'loss class "{loss_name}" not found, valid loss classes are: {list(loss_classes.keys())}'  # 如果没有找到指定的损失类，抛出断言错误
    return loss  # 返回损失类

def configure_optimization(pccnet, optim_config):  # 定义配置优化的函数
    """Configure the optimizers and the schedulers for training."""
    # 配置训练的优化器和调度器

    # Separate parameters for the main optimizer and the auxiliary optimizer
    # 为主优化器和辅助优化器分别设置参数
    parameters = set(
        n
        for n, p in pccnet.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    )
    aux_parameters = set(
        n
        for n, p in pccnet.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    )

    # Make sure we don't have an intersection of parameters
    # 确保我们没有参数的交集
    params_dict = dict(pccnet.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    # We only support the Adam optimizer to make things less complicated
    # 我们只支持Adam优化器，以使事情变得不那么复杂
    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(parameters))),
        lr=optim_config['main_args']['lr'],
        betas=(optim_config['main_args']['opt_args'][0], optim_config['main_args']['opt_args'][1]),
        weight_decay=optim_config['main_args']['opt_args'][2]
    )
    sche_args = optim_config['main_args']['schedule_args']
    if sche_args[0].lower() == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=sche_args[1])
    elif sche_args[0].lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sche_args[1], gamma=sche_args[2])
    elif sche_args[0].lower() == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=sche_args[1:-1], gamma=sche_args[-1])
    else: # 'fix' scheme
        scheduler = None

    # For the auxiliary parameters
    # 对于辅助参数
    if len(aux_parameters) > 0:
        aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(list(aux_parameters))),
            lr=optim_config['aux_args']['lr'],
            betas=(optim_config['aux_args']['opt_args'][0], optim_config['aux_args']['opt_args'][1]),
            weight_decay=optim_config['aux_args']['opt_args'][2]
        )
        aux_sche_args = optim_config['aux_args']['schedule_args']
        if aux_sche_args[0].lower() == 'exp':
                aux_scheduler = optim.lr_scheduler.ExponentialLR(aux_optimizer, gamma=aux_sche_args[1])
        elif aux_sche_args[0].lower() == 'step':
                aux_scheduler = optim.lr_scheduler.StepLR(aux_optimizer, step_size=aux_sche_args[1], gamma=aux_sche_args[2])
        elif aux_sche_args[0].lower() == 'multistep':
                aux_scheduler = optim.lr_scheduler.MultiStepLR(aux_optimizer, milestones=aux_sche_args[1:-1], gamma=aux_sche_args[-1])
        else: # 'fix' scheme
            aux_scheduler = None
    else:
        aux_optimizer = aux_scheduler = None

    return optimizer, scheduler, aux_optimizer, aux_scheduler
    # 这段代码主要用于配置网络优化的参数，包括优化器和调度器。其中，get_loss_class函数用于获取指定的损失类，configure_optimization函数用于配置优化参数，
    # 包括主优化器和辅助优化器的参数，以及对应的调度器。
