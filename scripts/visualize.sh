#!/usr/bin/env bash 该脚本可能用于可视化3D数据
 
# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Rendering settings
FILE="./datasets/ford/Ford_01_q_1mm/Ford_01_vox1mm-0100.ply"
RADIUS=-1
RADIUS_ORIGIN=-1
VIEW_FILE=.

# Begin rendering
python ./utils/visualize.py \
--file_name $FILE \
--output_file . \
--view_file $VIEW_FILE \
--radius $RADIUS \
--radius_origin $RADIUS_ORIGIN \
--window_name $FILE
# 这部分代码运行了visualize.py脚本，并将之前定义的变量作为参数传递给这个脚本。
# 这个脚本的功能可能是读取3D数据文件，并根据给定的参数生成一个可视化的窗口。
