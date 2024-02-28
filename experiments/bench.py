# 版权 (c) 2010-2022, InterDigital
# 保留所有权利。 

# 请参阅根文件夹下的 LICENSE。

# 对一个或多个模型进行基准测试

import multiprocessing
multiprocessing.set_start_method('spawn', True)  # 设置多进程的启动方式为 'spawn'

import random
import os
import torch
import sys
import csv
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')  # 将当前文件的父目录添加到系统路径

# 从 PccAI 加载不同的工具
from pccai.utils.option_handler import BenchmarkOptionHandler  # 加载基准测试选项处理器
import pccai.utils.logger as logger  # 加载日志工具
from pccai.pipelines.bench import *  # 加载基准测试管道


def aggregate_sequence_log(log_dict_all):
    '''
    将属于同一点云序列的字典聚合为一个字典，将在对动态点云序列进行基准测试时使用
    '''
    for ckpt in log_dict_all.keys():  # 遍历所有的字典
        log_dict_ckpt = log_dict_all[ckpt]
        log_dict_ckpt.sort(key=lambda x: x['seq_name'])  # 按照序列名称对字典进行排序
        cur_seq_name = ''
        log_dict_ckpt_aggregate=[]

        for idx, log_dict in enumerate(log_dict_ckpt):  # 遍历排序后的字典
            if log_dict['seq_name'].lower() != cur_seq_name:  # 遇到新的序列
                cur_seq_name = log_dict['seq_name'].lower()
                log_dict_tmp = {  # 创建一个新的字典，只包含 MPEG 报告所需的键
                    'pc_name': cur_seq_name,
                    'rec_num_points': log_dict['rec_num_points'],
                    'bit_total': log_dict['bit_total'],
                    'd1_psnr': log_dict['d1_psnr'],
                    'seq_cnt': 1
                }
                if 'd2_psnr' in log_dict:
                    log_dict_tmp['d2_psnr'] = log_dict['d2_psnr']
                if 'enc_time' in log_dict:
                    log_dict_tmp['enc_time'] = float(log_dict['enc_time'])
                if 'dec_time' in log_dict:
                    log_dict_tmp['dec_time'] = float(log_dict['dec_time'])
                log_dict_ckpt_aggregate.append(log_dict_tmp)
            else:  # 更新现有的序列
                log_dict_ckpt_aggregate[-1]['rec_num_points'] += log_dict['rec_num_points']
                log_dict_ckpt_aggregate[-1]['bit_total'] += log_dict['bit_total']
                log_dict_ckpt_aggregate[-1]['d1_psnr'] += log_dict['d1_psnr']
                log_dict_ckpt_aggregate[-1]['seq_cnt'] += 1
                if 'd2_psnr' in log_dict:
                    log_dict_ckpt_aggregate[-1]['d2_psnr'] += log_dict['d2_psnr']
                if 'enc_time' in log_dict:
                    log_dict_ckpt_aggregate[-1]['enc_time'] += float(log_dict['enc_time'])
                if 'dec_time' in log_dict:
                    log_dict_ckpt_aggregate[-1]['dec_time'] += float(log_dict['dec_time'])

        # 对每个序列取平均
        for idx, log_dict in enumerate(log_dict_ckpt_aggregate):
            log_dict['d1_psnr'] /= log_dict['seq_cnt']
            if 'd2_psnr' in log_dict:
                log_dict['d2_psnr'] /= log_dict['seq_cnt']
            if 'enc_time' in log_dict:
                log_dict['enc_time'] = str(log_dict['enc_time'])
            if 'dec_time' in log_dict:
                log_dict['dec_time'] = str(log_dict['dec_time'])

        log_dict_all[ckpt] = log_dict_ckpt_aggregate
    return None
# 这段代码的主要功能是对一个或多个模型进行基准测试。其中定义了一个函数 aggregate_sequence_log，该函数的作用是将属于同一点云序列的字典聚合为一个字典，
# 这在对动态点云序列进行基准测试时会用到。具体来说，它会遍历所有的字典，对每个字典按照序列名称进行排序，然后对每个序列进行处理：如果遇到新的序列，
# 就创建一个新的字典；如果遇到现有的序列，就更新该序列的字典。最后，对每个序列的字典进行平均处理。这样，我们就可以得到每个序列的平均基准测试结果。
# 这对于分析和比较不同模型的性能非常有用。


def flatten_ckpt_log(log_dict_all):
    '''
    原始的 log_dict_all 是一个由 ckpts 索引的字典，然后 log_dict_all[ckpt] 是一个包含几个字典的列表，
    每个字典对应一个推理测试的结果。这个函数将 log_dict_all 扁平化，所以输出的 log_dict_all_flat 是一个字典列表，
    并按照 pc_name（第一关键字）和 bit_total（第二关键字）进行排序
    '''
    log_dict_all_flat = []
    for ckpt, log_dict_ckpt in log_dict_all.items():
        for log_dict in log_dict_ckpt:
            log_dict['ckpt'] = ckpt
        log_dict_all_flat += log_dict_ckpt
    log_dict_all_flat.sort(key=lambda x: (x['pc_name'], int(x['bit_total'])))  # 使用两个关键字进行排序
    return log_dict_all_flat
    

def gen_mpeg_report(log_dict_all, mpeg_report_path, compute_d2, mpeg_report_sequence):
    """生成 MPEG 报告的 CSV 文件"""

    # 解析 MPEG 报告模板
    mpeg_seqname_file = os.path.join(os.path.split(__file__)[0], '..', 'assets', 'mpeg_test_seq.txt')
    with open(mpeg_seqname_file) as f:
        lines = f.readlines()
    mpeg_sequence_name = [str[:-1] for str in lines]

    # 对 log_dict_all 进行预处理
    if mpeg_report_sequence:
        aggregate_sequence_log(log_dict_all)
    log_dict_all = flatten_ckpt_log(log_dict_all)

    # 为 MPEG 报告写下 CSV 文件
    mpeg_report_dict_list = []
    for log_dict in log_dict_all:
        pc_name = os.path.splitext(log_dict['pc_name'])[0].lower()
        if pc_name[-2:] == '_n':
            pc_name = pc_name[:-2]
        if pc_name in mpeg_sequence_name:  # 找到一个 MPEG 序列
            mpeg_report_dict = {
                'sequence': pc_name,  # 序列
                'numOutputPointsT': log_dict['rec_num_points'],  # numOutputPointsT
                'numBitsGeoEncT': log_dict['bit_total'],  # numBitsGeoEncT
                'd1T': log_dict['d1_psnr']  # d1T,
            }
            if compute_d2:
                mpeg_report_dict['d2T'] = log_dict['d2_psnr']  # d2T

            # 编码/解码时间
            if 'enc_time' in log_dict:
                mpeg_report_dict['encTimeT'] = log_dict['enc_time']
            if 'dec_time' in log_dict:
                mpeg_report_dict['decTimeT'] = log_dict['dec_time']
            mpeg_report_dict_list.append(mpeg_report_dict)

    # 根据聚合的统计信息写 CSV 文件
    mpeg_report_header = ['sequence', 'numOutputPointsT', 'numBitsGeoEncT', 'd1T', 'd2T', 'encTimeT', 'decTimeT']
    with open(mpeg_report_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=mpeg_report_header)
        writer.writeheader()
        writer.writerows(mpeg_report_dict_list)
    if len(mpeg_report_dict_list) > 0:
        logger.log.info('CSV file for MPEG reporting: %s' % mpeg_report_path)
# 这段代码的主要功能是生成 MPEG 的报告 CSV 文件。其中定义了两个函数 flatten_ckpt_log 和 gen_mpeg_report。flatten_ckpt_log 函数的作用是将原始的 log_dict_all 字典扁平化，
# 输出的 log_dict_all_flat 是一个字典列表，并按照 pc_name（第一关键字）和 bit_total（第二关键字）进行排序。gen_mpeg_report 函数的作用是生成 MPEG 的报告 CSV 文件，
# 它首先解析 MPEG 报告模板，然后对 log_dict_all 进行预处理，接着为 MPEG 报告写下 CSV 文件，最后根据聚合的统计信息写 CSV 文件。这对于分析和比较不同模型的性能非常有用。
# 这段代码是用于点云压缩的基准测试，可以帮助我们理解和比较不同模型在压缩效率、压缩质量以及编码/解码时间等方面的性能。这对于选择合适的点云压缩模型非常有帮助。

if __name__ == "__main__":

    # 解析选项并进行训练
    option_handler = BenchmarkOptionHandler()  # 初始化基准测试选项处理器
    opt = option_handler.parse_options()  # 解析选项

    # 创建一个文件夹来保存模型和日志
    if not os.path.exists(opt.exp_folder):
        os.makedirs(opt.exp_folder)

    # 初始化一个全局日志，然后打印出所有的选项
    logger.create_logger(opt.exp_folder, opt.log_file, opt.log_file_only)  # 创建日志
    option_handler.print_options(opt)  # 打印选项
    opt = load_benchmark_config(opt)  # 加载基准测试配置

    # 进行实际的训练
    if opt.seed is not None:
        torch.manual_seed(opt.seed)  # 设置 PyTorch 的随机种子
        random.seed(opt.seed)  # 设置 Python 的随机种子
    log_dict_all = benchmark_checkpoints(opt)  # 对检查点进行基准测试

    # 如果需要，创建 MPEG 报告的 CSV 文件
    if opt.mpeg_report is not None:
        gen_mpeg_report(
            log_dict_all=log_dict_all, 
            mpeg_report_path=os.path.join(opt.exp_folder, opt.mpeg_report), 
            compute_d2=opt.compute_d2,
            mpeg_report_sequence=opt.mpeg_report_sequence
        )
    logger.log.info('Benchmarking session %s finished.\n' % opt.exp_name)  # 打印基准测试完成的信息
    logger.destroy_logger()  # 销毁日志
# 这段代码的主要功能是进行基准测试。首先，它会解析选项，然后创建一个文件夹来保存模型和日志，接着初始化一个全局日志并打印出所有的选项。
# 然后，它会加载基准测试配置，并进行实际的训练。在训练过程中，如果设置了随机种子，它会设置 PyTorch 和 Python 的随机种子。
# 然后，它会对检查点进行基准测试。最后，如果需要，它会创建 MPEG 报告的 CSV 文件，并打印基准测试完成的信息。
# 这段代码是用于点云压缩的基准测试，可以帮助我们理解和比较不同模型在压缩效率、压缩质量以及编码/解码时间等方面的性能。这对于选择合适的点云压缩模型非常有帮助。
