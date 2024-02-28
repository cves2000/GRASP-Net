#!/usr/bin/env bash

# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


if [ $# -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=$2
    echo export CUDA_VISIBLE_DEVICES=$2
fi
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
source './scripts/tmp/'$1
echo $1
echo python ${RUN_ARGUMENTS}
python ${RUN_ARGUMENTS}

# 这个.sh脚本文件是用来在特定的CUDA设备上运行Python脚本的。要使用这个文件，你需要按照以下步骤操作：
# 确保你的系统已经安装了Bash和Python环境，并且已经安装了CUDA。
# 打开终端，切换到脚本文件所在的目录。
# 给脚本文件添加执行权限。你可以使用chmod命令来做这件事，例如：
# chmod +x your_script.sh
# 这里，your_script.sh应该替换为你的脚本文件的名称。
# 运行脚本文件。你需要提供两个参数：第一个参数是你想要运行的Python脚本的名称，第二个参数是你想要使用的CUDA设备的编号。例如：
# ./your_script.sh your_python_script.py 0
# 这里，your_script.sh应该替换为你的脚本文件的名称，your_python_script.py应该替换为你想要运行的Python脚本的名称，0是你想要使用的CUDA设备的编号。
# 这个`.sh`脚本文件的主要用途是在特定的CUDA设备上运行Python脚本。具体来说，它做了以下几件事：
# 1. **设置CUDA设备**：脚本接受一个参数作为CUDA设备的编号，并将其设置为环境变量`CUDA_VISIBLE_DEVICES`。这意味着当你运行Python脚本时，CUDA只会使用你指定的设备。
# 2. **运行Python脚本**：脚本接受另一个参数作为Python脚本的名称，并运行这个脚本。这个Python脚本应该在`./scripts/tmp/`目录下。
# 3. **传递参数给Python脚本**：脚本从`./scripts/tmp/`目录下的一个文件中读取参数，并将这些参数传递给Python脚本。这个文件的名称是脚本的第一个参数。
# 总的来说，这个脚本的作用是让你能够方便地在特定的CUDA设备上运行Python脚本，并向这个脚本传递参数。这对于进行大规模的机器学习或深度学习训练非常有用，因为这些任务通常需要在特定的GPU上运行，并需要大量的参数设置。
