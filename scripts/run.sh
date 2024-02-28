#!/usr/bin/env bash 这是一个shebang行，它告诉系统这个脚本应该使用哪个解释器来执行。在这里，/usr/bin/env bash表示这个脚本应该使用Bash来执行。

# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


SPEC=$1
LAUNCHER=$2
USE_GPU=$3
#这部分代码定义了三个变量，它们的值来自于脚本的参数。SPEC是第一个参数，LAUNCHER是第二个参数，USE_GPU是第三个参数。

TMP_ARGS=`python ./utils/gen_args.py ${SPEC}`#这行代码运行了一个Python脚本./utils/gen_args.py，并将其输出赋值给TMP_ARGS变量。这个Python脚本的参数是SPEC变量的值。
if [[ ${LAUNCHER} == "d" ]]; then
    echo "Launch the job directly."
    ./scripts/config_args.sh ${TMP_ARGS} ${USE_GPU} 2>&1 &
elif [[ ${LAUNCHER} == "f" ]]; then
    echo "Launch the job directly in foreground."
    ./scripts/config_args.sh ${TMP_ARGS} ${USE_GPU} 2>&1
elif [[ ${LAUNCHER} == "s" ]]; then
    echo "Launch the job with slurm."
    source './scripts/tmp/'${TMP_ARGS}
    # Please modify according your needs
    sbatch --job-name=${EXP_NAME} -n 1 -D ${HOME_DIR} --gres=gpu:1 ./scripts/config_args.sh ${TMP_ARGS} 0
else
    echo "No launcher is specified."
fi
# 这部分代码根据LAUNCHER变量的值来决定如何运行另一个脚本./scripts/config_args.sh。
# 如果LAUNCHER的值是"d"，那么就在后台直接运行这个脚本。
# 如果LAUNCHER的值是"f"，那么就在前台直接运行这个脚本。
# 如果LAUNCHER的值是"s"，那么就使用Slurm作业调度器来运行这个脚本。
# 如果LAUNCHER的值是其他的值，那么就打印出"No launcher is specified."。
# 这个Bash脚本的主要用途是在特定的CUDA设备上运行Python脚本，并且可以选择在后台、前台或使用Slurm作业调度器来运行。具体来说，它做了以下几件事：
# 解析输入参数：脚本接受三个参数。第一个参数SPEC是你想要运行的Python脚本的名称，第二个参数LAUNCHER决定了如何运行Python脚本（直接在后台运行、直接在前台运行或使用Slurm作业调度器运行），第三个参数USE_GPU是你想要使用的CUDA设备的编号。
# 生成运行参数：脚本运行./utils/gen_args.py Python脚本，并将其输出作为运行Python脚本的参数。
# 运行Python脚本：根据LAUNCHER变量的值来决定如何运行另一个脚本./scripts/config_args.sh。这个脚本负责设置环境变量并运行Python脚本。
# 总的来说，这个脚本的作用是让你能够方便地在特定的CUDA设备上运行Python脚本，并向这个脚本传递参数。
