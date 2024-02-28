# 版权所有 (c) 2010-2022，InterDigital
# 保留所有权利。
# 请在根文件夹下查看LICENSE。

# 一个用于将所有信息输出到显示器和指定文件的记录器

import logging  # 导入日志库
import sys  # 导入系统库
import os  # 导入操作系统库

log = None  # 初始化日志为空


def create_logger(exp_folder, file_name, log_file_only):  # 定义创建记录器的函数

    if log_file_only:  # 如果只记录到文件
        handlers = []  # 处理器为空
    else:  # 否则
        handlers = [logging.StreamHandler(sys.stdout)]  # 处理器为标准输出
    if file_name != '':  # 如果文件名不为空
        log_path = os.path.join(exp_folder, file_name)  # 创建日志路径
        os.makedirs(os.path.split(log_path)[0], exist_ok=True)  # 创建日志路径的目录
        handlers.append(logging.FileHandler(log_path, mode = 'w'))  # 添加文件处理器
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]] # 删除所有现有的处理器
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', handlers=handlers)  # 配置基本日志
    global log  # 声明全局变量log
    log = logging.getLogger()  # 获取日志

def destroy_logger():  # 定义销毁记录器的函数
    handlers = log.handlers[:]  # 获取处理器
    for handler in handlers:  # 对于每一个处理器
        handler.close()  # 关闭处理器
        log.removeHandler(handler)  # 移除处理器
# 这段代码主要用于创建和销毁记录器，记录器用于将所有信息输出到显示器和指定的文件。其中，create_logger函数用于创建记录器，destroy_logger函数用于销毁记录器。
