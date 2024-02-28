# 版权所有 (c) 2010-2022，InterDigital
# 保留所有权利。
# 请在根文件夹下查看LICENSE。

# 在训练、测试、基准测试等过程中处理所有输入参数。

import pccai.utils.logger as logger  # 导入pccai.utils.logger作为logger
import argparse  # 导入argparse库
import os  # 导入os库

def str2bool(val):  # 定义str2bool函数
    if isinstance(val, bool):  # 如果val是布尔值
        return val  # 返回val
    if val.lower() in ('true', 'yes', 't', 'y', '1'):  # 如果val在这些值中
        return True  # 返回True
    elif val.lower() in ('false', 'no', 'f', 'n', '0'):  # 如果val在这些值中
        return False  # 返回False
    else:  # 否则
        raise argparse.ArgumentTypeError('Expect a Boolean value.')  # 抛出异常


class BasicOptionHandler():  # 定义BasicOptionHandler类
    """包含所有阶段共享的基本选项的类。"""

    def __init__(self):  # 初始化函数
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 创建解析器
        parser = self.add_options(parser)  # 添加选项
        self.parser = parser  # 设置解析器

    def add_options(self, parser):  # 定义添加选项的函数

        # 你想做什么
        parser.add_argument('--exp_name', type=str, default='experiment_name', help='实验的名称，根据此名称创建结果文件夹。')

        # 你的原料是什么
        parser.add_argument('--net_config', type=str, default='', help='YAML中的网络配置。')
        parser.add_argument('--optim_config', type=str, default='', help='YAML中的优化配置。')
        parser.add_argument('--hetero', type=str2bool, nargs='?', const=True, default=False, help='是否使用异构批处理模式。')
        parser.add_argument('--checkpoint', type=str, default='', help='加载现有的检查点。')

        # 你怎么做
        parser.add_argument('--alpha', type=float, default=None, help='R-D优化中的失真权重，可以覆盖YAML配置中的值。')
        parser.add_argument('--beta', type=float, default=None, help='R-D优化中的比特率权重，可以覆盖YAML配置中的值。')
        parser.add_argument('--seed', type=float, default=None, help='设置随机种子以实现可重复性')

        # 记录选项
        parser.add_argument('--result_folder', type=str, default='results', help='指定结果文件夹。')
        parser.add_argument('--log_file', type=str, default='', help='日志文件名。')
        parser.add_argument('--log_file_only', type=str2bool, nargs='?', const=True, default=False, help='如果设置为True，只打印到日志文件。')
        parser.add_argument('--print_freq', type=int, default=20, help='显示结果的频率。')
        parser.add_argument('--pc_write_freq', type=int, default=50, help='写下点云的频率，使用tensorboard在训练期间写入，测试期间写入"ply"文件proint cloud。')  
        parser.add_argument('--tf_summary', type=str2bool, nargs='?', const=True, default=False, help='是否使用tensorboard进行日志。')
        return parser  # 返回解析器

    def parse_options(self):  # 定义解析选项的函数
        opt, _ = self.parser.parse_known_args()  # 解析已知参数
        opt.exp_folder = os.path.join(opt.result_folder, opt.exp_name)  # 设置实验文件夹
        return opt  # 返回选项

    def print_options(self, opt):  # 定义打印选项的函数
        message = ''  # 初始化消息为空
        message += '\\n----------------- 输入参数 ---------------\\n'
        # For k, v in sorted(vars(opt).items()):
        for k, v in vars(opt).items():  # 对于选项中的每一项
            comment = ''  # 初始化注释为空
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        logger.log.info(message)
# 这段代码主要用于在训练、测试、基准测试等过程中处理所有输入参数。
# 其中，str2bool函数用于将字符串转换为布尔值，BasicOptionHandler类用于处理所有阶段共享的基本选项，包括添加选项、解析选项和打印选项。



class TrainOptionHandler(BasicOptionHandler):  # 定义TrainOptionHandler类，包含训练的特定选项
    """包含训练的特定选项的类。"""

    def add_options(self, parser):  # 定义添加选项的函数
        parser = BasicOptionHandler.add_options(self, parser)  # 添加基本选项
        parser.add_argument('--train_data_config', type=str, nargs='+', required=True, help='YAML中的训练数据配置。')
        parser.add_argument('--val_data_config', type=str, nargs='+', default='', help='YAML中的验证数据配置。')
        parser.add_argument('--checkpoint_optim_config', type=str2bool, nargs='?', const=True, default=False, help='是否从检查点加载优化器和调度器。')
        parser.add_argument('--checkpoint_epoch_state', type=str2bool, nargs='?', const=True, default=False, help='是否从检查点加载epoch状态。')
        parser.add_argument('--save_checkpoint_freq', type=int, default=2, help='保存训练模型的频率。')
        parser.add_argument('--save_checkpoint_max', type=int, default=10, help='保存的检查点的最大数量。')
        parser.add_argument('--save_checkpoint_prefix', type=str, default='epoch_', help='检查点文件名的前缀。')
        parser.add_argument('--val_freq', type=int, default=-1, help='使用验证集进行验证的频率，<=0表示不进行验证。')
        parser.add_argument('--val_print_freq', type=int, default=20, help='在验证期间显示结果的频率。')
        parser.add_argument('--lr', type=float, default=None, help='主参数的学习率，可以覆盖YAML配置中的值。')
        parser.add_argument('--lr_aux', type=float, default=None, help='辅助参数的学习率，可以覆盖YAML配置中的值。')
        parser.add_argument('--fix_modules', type=str, nargs='+', default='', help='在训练期间固定的模块的名称。')
        parser.add_argument('--ddp', type=str2bool, nargs='?', const=True, default=False, help='是否使用DPP模式。')
        parser.add_argument('--master_address', type=str, default='localhost', help='DDP的主地址。')
        parser.add_argument('--master_port', type=int, default=29500, help='DDP的主端口。')

        # 方法特定的参数
        parser.add_argument('--scaling_ratio', type=float, required=True, help='[GRASP] 缩放比例。')
        parser.add_argument('--point_mul', type=int, required=True, help='[GRASP] 点乘参数。')
        parser.add_argument('--skip_mode',  type=str2bool, nargs='?', required=True, const=True, default=False, help='[GRASP] 是否跳过模式。')

        # 如果需要，你可以在这里添加你的方法特定的参数。他们可以在训练前传递给加载的YAML配置。
        # 查看pipelines/train.py中如何覆盖alpha、beta和lr作为示例。
        return parser


class TestOptionHandler(BasicOptionHandler):  # 定义TestOptionHandler类，包含测试的特定选项
    """包含测试的特定选项的类。"""

    def add_options(self, parser):  # 定义添加选项的函数
        parser = BasicOptionHandler.add_options(self, parser)  # 添加基本选项
        parser.add_argument('--checkpoint_net_config', type=str2bool, nargs='?', const=True, default=False, help='是否从检查点加载模型配置，如果是，将忽略net_config。')
        parser.add_argument('--test_data_config', type=str, nargs='+', required=True, help='YAML中的测试数据配置。')
        parser.add_argument('--gen_bitstream', type=str2bool, nargs='?', const=True, default=False, help='是否生成实际的比特流。')
        parser.add_argument('--pc_write_prefix', type=str, default='', help='写下点云时的前缀。')  
        return parser
# 这段代码主要用于在训练和测试过程中处理所有输入参数。其中，TrainOptionHandler类用于处理训练的特定选项，TestOptionHandler类用于处理测试的特定选项。

class BenchmarkOptionHandler(BasicOptionHandler):  # 定义BenchmarkOptionHandler类，包含基准测试的特定选项
    """包含基准测试的特定选项的类。"""

    def add_options(self, parser):  # 定义添加选项的函数
        parser = BasicOptionHandler.add_options(self, parser)  # 添加基本选项
        parser.add_argument('--checkpoints', type=str, nargs='+', default=None, help='指定几个现有的检查点。')
        parser.add_argument('--checkpoint_net_config', type=str2bool, nargs='?', const=True, default=True, help='是否从检查点加载模型配置，如果是，将忽略net_config。')
        parser.add_argument('--codec_config', type=str, required=True, help='YAML中的编解码器配置。')
        parser.add_argument('--input', type=str, nargs='+', required=True, help='包含要测试的点云的文件夹列表，或者只是一个ply文件。')
        parser.add_argument('--peak_value', type=int, nargs='+', required=True, help='计算D1和D2度量的峰值。如果只提供了一个值，它将用于整个测试；否则需要为每个点云给出峰值。')
        parser.add_argument('--bit_depth', type=int, nargs='+', required=True, help='要测试的点云的位深值。如果只提供了一个值，它将用于整个测试；否则需要为每个点云给出位深。')
        parser.add_argument('--remove_compressed_files', type=str2bool, nargs='?', const=True, default=True, help='是否删除压缩文件。')
        parser.add_argument('--skip_decode', type=str2bool, nargs='?', const=True, default=False, help='是否跳过解码过程，对于无损压缩很有用。')
        parser.add_argument('--compute_d2', type=str2bool, nargs='?', const=True, default=False, help='是否计算D2度量。')
        parser.add_argument('--mpeg_report', type=str, default=None, help='以CSV格式写入MPEG报告的结果。')
        parser.add_argument('--mpeg_report_sequence', type=str2bool, nargs='?', const=True, default=False, help='如果为true，通过将输入视为点云序列，以CSV格式创建MPEG报告。')
        parser.add_argument('--write_prefix', type=str, default='', help='写下点云和比特流时的前缀。')
        parser.add_argument('--slice', type=int, default=None, help='切片参数。')
        return parser
# 这段代码主要用于在基准测试过程中处理所有输入参数。其中，BenchmarkOptionHandler类用于处理基准测试的特定选项。
