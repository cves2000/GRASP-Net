# Copyright (c) 2010-2022, InterDigital
# 版权所有 (c) 2010-2022，InterDigital
# All rights reserved. 
# 保留所有权利。
# See LICENSE under the root folder.
# 请在根文件夹下查看LICENSE。

# Convert a LiDAR point cloud to a range image based on spherical coordinate conversion
# 基于球坐标转换，将LiDAR点云转换为范围图像

import numpy as np  # 导入NumPy库


def cart2spherical(input_xyz):  # 定义从笛卡尔坐标到球坐标的转换函数
    """Conversion from Cartisian coordinates to spherical coordinates."""
    # 从笛卡尔坐标转换到球坐标

    r = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2 + input_xyz[:, 2] ** 2)  # 计算半径r
    alpha = np.arctan2(input_xyz[:, 1], input_xyz[:, 0]) # corresponding to width  # 计算宽度对应的角度alpha
    epsilon = np.arcsin(input_xyz[:, 2] / r) # corrsponding to height  # 计算高度对应的角度epsilon
    return np.stack((r, alpha, epsilon), axis = 1)  # 返回堆叠的结果


def spherical2cart(input_spherical):  # 定义从球坐标到笛卡尔坐标的转换函数
    """Conversion from spherical coordinates to Cartesian coordinates."""
    # 从球坐标转换到笛卡尔坐标

    x = input_spherical[:, 0] * np.cos(input_spherical[:, 1]) * np.cos(input_spherical[:, 2])  # 计算x坐标
    y = input_spherical[:, 0] * np.sin(input_spherical[:, 1]) * np.cos(input_spherical[:, 2])  # 计算y坐标
    z = input_spherical[:, 0] * np.sin(input_spherical[:, 2])  # 计算z坐标
    return np.stack((x, y, z), axis=1)  # 返回堆叠的结果


def pc2img(h_fov, v_fov, width, height, inf, data):  # 定义从点云到图像的转换函数
    """Convert a point cloud to an 2D image."""
    # 将点云转换为2D图像

    data_spherical = cart2spherical(data)  # 将点云数据从笛卡尔坐标转换为球坐标

    # Project the point cloud onto an image!
    # 将点云投影到图像上！
    x = (data_spherical[:, 1] - h_fov[0]) / (h_fov[1] - h_fov[0])  # 计算x坐标
    y = (data_spherical[:, 2] - v_fov[0]) / (v_fov[1] - v_fov[0])  # 计算y坐标
    x = np.round(x * (width - 1)).astype(np.int32)  # 将x坐标四舍五入并转换为整数
    y = np.round(y * (height - 1)).astype(np.int32)  # 将y坐标四舍五入并转换为整数

    # exclude the pixels that are out of the selected FOV
    # 排除超出选定视场的像素
    mask = ~((x < 0) | (x >= width) | (y < 0) | (y >= height))  # 创建掩码
    x, y = x[mask], y[mask]  # 应用掩码
    range = data_spherical[:, 0][mask]  # 获取范围
    data_img = np.ones((height, width), dtype = np.float32) * inf  # 创建图像数据
    data_img[y, x] = range  # 将范围写入图像数据

    return data_img  # 返回图像数据


def img2pc(h_fov, v_fov, width, height, inf, data):  # 定义从图像到点云的转换函数
    """Convert an 2D image back to the point cloud."""
    # 将2D图像转换回点云
    
    alpha = (np.arange(width) / (width - 1)) * (h_fov[1] - h_fov[0]) + h_fov[0]  # 计算alpha
    epsilon = (np.arange(height) / (height - 1)) * (v_fov[1] - v_fov[0]) + v_fov[0]  # 计算epsilon
    alpha, epsilon = np.meshgrid(alpha, epsilon)  # 创建网格
    data_pc = np.stack((data, alpha, epsilon), axis=2)  # 堆叠数据
    data_pc = data_pc.reshape(-1, 3)  # 重塑数据
    data_pc = data_pc[data_pc[:, 0] < inf - 1, :]  # 获取数据
    data_pc = spherical2cart(data_pc)  # 将数据从球坐标转换为笛卡尔坐标

    return data_pc  # 返回点云
    # 这段代码主要用于将LiDAR点云转换为范围图像，以及将范围图像转换回点云。其中，cart2spherical函数用于将笛卡尔坐标转换为球坐标，
    # spherical2cart函数用于将球坐标转换为笛卡尔坐标，pc2img函数用于将点云转换为图像，img2pc函数用于将图像转换回点云
