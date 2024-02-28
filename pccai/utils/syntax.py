# 版权所有 (c) 2010-2022，InterDigital
# 保留所有权利。
# 请在根文件夹下查看LICENSE。

# 定义并生成内部语法和状态，服务于异构模式并标记模块阶段

def gen_syntax_gt(hetero):  # 定义生成语法的函数
    if hetero:  # 如果是异构模式
        syntax_gt = {  # 定义语法
            '__len__': 10,
            'xyz': [0, 2],
            'block_pntcnt': 3,
            'block_center': [4, 6],
            'block_scale': 7,
            'block_start': 9,
        }
    else:  # 如果不是异构模式
        syntax_gt = None  # 语法为空
    return syntax_gt  # 返回语法


class SyntaxGenerator():  # 定义语法生成器类
    """生成内部数据和模块状态通信的语法。"""

    def __init__(self, opt):  # 初始化函数
        self.hetero = opt.hetero  # 设置异构模式
        self.phase = opt.phase  # 设置阶段
        self.generate_syntax_gt()  # 生成语法
        self.generate_syntax_rec()  # 生成语法
        self.generate_syntax_cw(opt.net_config)  # 生成语法

    def generate_syntax_gt(self, **kwargs):  # 定义生成语法的函数
        """xyz必须排在前面，其余的可以交换
        数据语法：x, y, z, block_pntcnt, block_center, block_scale, block_start
        索引：0, 1, 2,      3,         4 ~ 6,           7,           8
        """
        self.syntax_gt = gen_syntax_gt(self.hetero)  # 生成语法

    def generate_syntax_rec(self, **kwargs):  # 定义生成语法的函数
        """xyz必须排在前面，其余的可以交换
        推荐语法：x, y, z, pc_start
        索引：0, 1, 2,     3
        """
        if self.hetero:  # 如果是异构模式
            self.syntax_rec = {  # 定义语法
                '__len__': 10,
                'xyz': [0, 2],
                'block_start': 3,
                'block_center': [4, 6],
                'block_scale': 7,
                'pc_start': 8,
            }
        else: self.syntax_rec = None  # 如果不是异构模式，语法为空

    def generate_syntax_cw(self, net_config, **kwargs):  # 定义生成语法的函数
        """编码字必须排在前面，其余的可以交换
        代码语法：codeword, block_pntcnt, block_center, block_scale, pc_start
        索引：  0 ~ 511,     512,        513 ~ 515,      516,        517
                            \--------------------  --------------------/
                                                        \/
                                                    meta_data
        """
        if self.hetero:  # 如果是异构模式
            len_cw = net_config['modules']['entropy_bottleneck']  # 获取编码字长度
            self.syntax_cw = {  # 定义语法
                '__len__': len_cw + 7,
                '__meta_idx__': len_cw,
                'cw': [0, len_cw - 1],
                'block_pntcnt': len_cw,
                'block_center': [len_cw + 1, len_cw + 3],
                'block_scale': len_cw + 4,
                'pc_start': len_cw + 5,
            }
        else: self.syntax_cw = None  # 如果不是异构模式，语法为空


def syn_slc(syntax, attr):  # 定义创建切片的函数
    """从语法和键创建切片"""

    syn = syntax[attr]  # 获取语法
    return slice(syn[0], syn[1] + 1)  # 返回切片
    # 用于定义并生成内部语法和状态，服务于异构模式并标记模块阶段。其中，gen_syntax_gt函数用于生成语法，
    # SyntaxGenerator类用于生成内部数据和模块状态通信的语法，syn_slc函数用于从语法和键创建切片。
