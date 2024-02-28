# 版权所有 (c) 2010-2022，InterDigital
# 保留所有权利。
# 请在根文件夹下查看LICENSE。

import subprocess  # 导入子进程库
import os  # 导入操作系统库
import random  # 导入随机库

from pccai.utils.misc import pc_write  # 从pccai.utils.misc导入pc_write函数
base_path = os.path.split(__file__)[0]  # 获取文件的基础路径

def compute_metrics(gt_file, pc_rec, res, normal=False):  # 定义计算度量的函数
    """使用MPEG的pc_error工具计算D1和/或D2"""

    tmp_file_name = os.path.join('./tmp/', 'metric_'+str(hex(int(random.random() * 1e15)))+'.ply')  # 创建临时文件名
    rec_file = os.path.join(base_path, '../..', tmp_file_name)  # 创建记录文件的路径
    pc_error_path = os.path.join(base_path, '../..', 'third_party/pc_error')  # 创建pc_error的路径
    pc_write(pc_rec, rec_file)  # 写入点云记录
    cmd = pc_error_path + ' -a '+ gt_file + ' -b '+ rec_file + ' --hausdorff=1 '+ ' --resolution=' + str(res)  # 创建命令
    if normal: cmd = cmd + ' -n ' + gt_file  # 如果normal为True，添加'-n'和gt_file到命令
    bg_proc=subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)  # 执行命令
    line_b = bg_proc.stdout.readline()  # 读取输出的一行

    d1_key = 'mseF,PSNR (p2point):'  # 定义d1_key
    d2_key = 'mseF,PSNR (p2plane):'  # 定义d2_key
    d1_psnr, d2_psnr = None, None  # 初始化d1_psnr和d2_psnr为None
    while line_b:  # 当line_b存在时
        line = line_b.decode(encoding='utf-8')  # 解码line_b
        line_b = bg_proc.stdout.readline()  # 读取输出的一行
        idx = line.find(d1_key)  # 查找d1_key在line中的位置
        if idx > 0: d1_psnr = float(line[idx + len(d1_key):])  # 如果找到d1_key，获取d1_psnr
        if normal:  # 如果normal为True
            idx = line.find(d2_key)  # 查找d2_key在line中的位置
            if idx > 0: d2_psnr = float(line[idx + len(d2_key):])  # 如果找到d2_key，获取d2_psnr
    os.remove(rec_file)  # 删除记录文件
    return {"d1_psnr": d1_psnr, "d2_psnr": d2_psnr}  # 返回d1_psnr和d2_psnr
# 这段代码主要用于计算D1和/或D2，这是通过使用MPEG的pc_error工具来实现的。其中，compute_metrics函数用于计算度量，
# 它首先创建一个临时文件名和记录文件的路径，然后执行一个命令来计算D1和D2。
