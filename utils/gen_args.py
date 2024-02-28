# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# A simple tool to generate temporary scripts which holds the options 

# 导入所需的模块
import sys
import os
import random

# 获取当前文件的绝对路径和上级目录
cur_dir = os.path.dirname(os.path.abspath(__file__))
home_dir=os.path.abspath(os.path.join(cur_dir, '..'))

def main():
    # 定义临时脚本文件夹的名称
    tmp_script_folder = 'tmp'
    # 获取临时脚本文件夹的路径
    tmp_script_path = os.path.join(home_dir, 'scripts', tmp_script_folder)
    # 如果临时脚本文件夹不存在，则创建该文件夹
    if not os.path.exists(tmp_script_path):
        os.makedirs(tmp_script_path)

    # 生成一个随机的临时文件名，并在指定的路径下创建这个文件
    tmp_file_name = 'tmp_'+str(hex(int(random.random() * 1e15)))+'.sh'
    tmp_file = open(os.path.join(home_dir, 'scripts', 'tmp', tmp_file_name), 'w')

    # 向临时文件中写入两个变量：HOME_DIR和EXP_NAME
    tmp_file.write('HOME_DIR="' + home_dir + '"\n')
    exp_name = os.path.basename(sys.argv[1]).split('.')[0]
    tmp_file.write('EXP_NAME="' + exp_name + '"\n')

    # 从另一个文件中读取参数，并将这些参数添加到临时文件中
    addline = 'RUN_ARGUMENTS="${PY_NAME} --exp_name ${EXP_NAME} '
    len_addline = len(addline)
    with open(sys.argv[1]) as f:
        args = f.readlines()
        for line in args:
            line = line.lstrip()
            if len(line) > 0 and line[0].isalpha():
                idx = line.find('=')
                opt_name = line[0:idx].upper()
                if opt_name != "PY_NAME" and opt_name != "EXP_NAME":
                    addline += "--" + opt_name.lower() + " ${" + opt_name + "} "
                if opt_name != 'RUN_ARGUMENTS' and opt_name != "EXP_NAME":
                    tmp_file.write(line)
        addline = "\n" + addline[:-1] + '"'
    if len(addline) > len_addline:
        tmp_file.write(addline)

    # 主函数返回临时文件的名称
    return tmp_file_name
    

# 检查当前脚本是否作为主程序运行，如果是，那么就调用主函数，并打印出生成的临时文件的名称。然后，程序退出。
if __name__ == "__main__":
    tmp_file_name = main()
    print(tmp_file_name)
    sys.exit(0)
