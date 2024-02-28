# 版权所有 (c) 2010-2022，InterDigital
# 保留所有权利。
# 请在根文件夹下查看LICENSE。

# 使用广度优先搜索进行八叉树划分和去划分

import os  # 导入操作系统库
import pickle  # 导入pickle库
import numpy as np  # 导入NumPy库

def compute_new_bbox(idx, bbox_min, bbox_max):  # 定义计算新边界框的函数
    """给定索引，计算全局块边界框。"""

    midpoint = (bbox_min + bbox_max) / 2  # 计算中点
    cur_bbox_min = bbox_min.copy()  # 复制bbox_min
    cur_bbox_max = midpoint.copy()  # 复制midpoint
    if idx & 1:  # 如果idx和1进行位与运算的结果为真
        cur_bbox_min[0] = midpoint[0]  # 设置cur_bbox_min的第0个元素为midpoint的第0个元素
        cur_bbox_max[0] = bbox_max[0]  # 设置cur_bbox_max的第0个元素为bbox_max的第0个元素
    if (idx >> 1) & 1:  # 如果idx右移1位后和1进行位与运算的结果为真
        cur_bbox_min[1] = midpoint[1]  # 设置cur_bbox_min的第1个元素为midpoint的第1个元素
        cur_bbox_max[1] = bbox_max[1]  # 设置cur_bbox_max的第1个元素为bbox_max的第1个元素
    if (idx >> 2) & 1:  # 如果idx右移2位后和1进行位与运算的结果为真
        cur_bbox_min[2] = midpoint[2]  # 设置cur_bbox_min的第2个元素为midpoint的第2个元素
        cur_bbox_max[2] = bbox_max[2]  # 设置cur_bbox_max的第2个元素为bbox_max的第2个元素

    return cur_bbox_min, cur_bbox_max  # 返回cur_bbox_min和cur_bbox_max

def _analyze_octant(points, bbox_min, bbox_max):  # 定义分析八分体的函数
    """分析给定块中点的统计信息。"""

    center = (np.asarray(bbox_min) + np.asarray(bbox_max)) / 2  # 计算中心

    locations = (points >= np.expand_dims(center, 0)).astype(np.uint8)  # 计算位置
    locations *= np.array([[1, 2, 4]], dtype=np.uint8)  # 位置乘以数组
    locations = np.sum(locations, axis=1)  # 计算位置的和

    location_cnt = np.zeros((8,), dtype=np.uint32)  # 初始化位置计数为零
    for idx in range(locations.shape[0]):  # 对于每一个位置
        loc = locations[idx]  # 获取位置
        location_cnt[loc] += 1  # 增加位置计数

    location_map = np.zeros(locations.shape[0], dtype=np.uint32)  # 初始化位置映射为零
    location_idx = np.zeros((8,), dtype=np.uint32)  # 初始化位置索引为零
    for i in range(1, location_idx.shape[0]):  # 对于每一个位置索引
        location_idx[i] = location_idx[i-1] + location_cnt[i-1]  # 更新位置索引
    for idx in range(locations.shape[0]):  # 对于每一个位置
        loc = locations[idx]  # 获取位置
        location_map[location_idx[loc]] = idx  # 更新位置映射
        location_idx[loc] += 1  # 增加位置索引

    # 当前节点的占用模式
    pattern = np.sum((location_cnt > 0).astype(np.uint32) * np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint32))
    points = points[location_map, :] # 重新排列点
    child_bboxes = [compute_new_bbox(i, bbox_min, bbox_max) for i in range(8)]  # 计算子边界框

    return points, location_cnt, pattern, child_bboxes, location_map  # 返回点、位置计数、模式、子边界框和位置映射

def analyze_octant(points, bbox_min, bbox_max, attr=None):  # 定义分析八分体的函数
    points, location_cnt, pattern, child_bboxes, location_map = _analyze_octant(points, bbox_min, bbox_max)  # 分析八分体
    if attr is not None:  # 如果attr不为空
        attr = attr[location_map, :]  # 更新attr
    
    return points, location_cnt, pattern, child_bboxes, attr  # 返回点、位置计数、模式、子边界框和attr


# 这段代码主要用于处理八叉树的划分和去划分，使用的是广度优先搜索。
# 其中，compute_new_bbox函数用于计算新的边界框，_analyze_octant函数用于分析给定块中点的统计信息，analyze_octant函数用于分析八分体。
class OctreeConverter():
    """
    一个用于存储八叉树参数并执行八叉树划分的类。
    """

    def __init__(self, bbox_min, bbox_max, point_min, point_max, level_min, level_max):
    
        # 设置八叉树划分选项
        self.bbox_min, self.bbox_max = np.asarray(bbox_min, dtype=np.float32), np.asarray(bbox_max, dtype=np.float32)
        self.point_min, self.point_max = point_min, point_max
        self.level_min, self.level_max = level_min, level_max
        self.normalized_box_size = 2


    def leaf_test(self, point_cnt, level):
        """确定一个块是否是叶子。"""
        return (level >= self.level_max) or (point_cnt <= self.point_max and level >= self.level_min)


    def skip_test(self, point_cnt):
        """确定是否应跳过一个块。"""
        return point_cnt < self.point_min # True: 跳过; False: 转换


    def partition_octree(self, points, attr=None):
        """使用广度优先搜索进行八叉树划分。"""

        # 删除超出边界框的点
        mask = np.ones(points.shape[0], dtype=bool)
        for i in range(3):
            mask = mask & (points[:, i] >= self.bbox_min[i]) & (points[:, i] <= self.bbox_max[i])
        points = points[mask,:]
        if attr is not None: attr = attr[mask,:]

        # 初始化
        root_block = {'level': 0, 'bbox_min': self.bbox_min, 'bbox_max': self.bbox_max, 'pnt_range': np.array([0, points.shape[0] - 1]), 'parent': -1, 'binstr': 0}
        blocks = [root_block]
        leaf_idx = []
        cur = 0

        # 开始划分
        while True:
            pnt_start, pnt_end = blocks[cur]['pnt_range'][0], blocks[cur]['pnt_range'][1]
            point_cnt = pnt_end - pnt_start + 1
            if self.leaf_test(point_cnt, blocks[cur]['level']): # 找到一个叶节点
                leaf_idx.append(cur)
                if self.skip_test(point_cnt): # 如果点很少，使用跳过转换
                    blocks[cur]['binstr'] = -1 # -1 - "跳过"; 0 - "转换"
            else: # 划分当前节点
                points[pnt_start : pnt_end + 1], location_cnt, blocks[cur]['binstr'], child_bboxes, attr_tmp = \
                    analyze_octant(points[pnt_start : pnt_end + 1], blocks[cur]['bbox_min'], blocks[cur]['bbox_max'],
                    attr[pnt_start : pnt_end + 1] if attr is not None else None)
                if attr is not None: attr[pnt_start : pnt_end + 1] = attr_tmp

                # 创建子节点            
                location_idx = np.insert(np.cumsum(location_cnt, dtype=np.uint32), 0, 0) + blocks[cur]['pnt_range'][0]
                for idx in range(8):
                    if location_cnt[idx] > 0: # 如果还有点，创建一个子节点
                        block = {'level': blocks[cur]['level'] + 1, 'bbox_min': child_bboxes[idx][0], 'bbox_max': child_bboxes[idx][1],
                            'pnt_range': np.array([location_idx[idx], location_idx[idx + 1] - 1], dtype=location_idx.dtype),
                            'parent': cur, 'binstr': 0}
                        blocks.append(block)
            cur += 1
            if cur >= len(blocks): break

        binstrs = np.asarray([np.max((blocks[i]['binstr'], 0)) for i in range(len(blocks))]).astype(np.uint8) # 最终的二进制字符串总是大于等于0
        return blocks, leaf_idx, points, attr, binstrs
    # 这段代码主要用于处理八叉树的划分。其中，OctreeConverter类用于存储八叉树参数并执行八叉树划分，leaf_test函数用于确定一个块是否是叶子，skip_test函数用于确定是否应跳过一个块，
    # partition_octree函数用于使用广度优先搜索进行八叉树划分。


def departition_octree(self, binstrs, block_pntcnt):
        """使用广度优先搜索对给定的八叉树进行去划分。
        给定二进制字符串和边界框，恢复每个叶节点的边界框和级别。
        """

        # 初始化
        root_block = {'level': 0, 'bbox_min': self.bbox_min, 'bbox_max': self.bbox_max}
        blocks = [root_block]
        leaf_idx = []
        cur = 0

        while True:
            blocks[cur]['binstr'] = binstrs[cur]
            if blocks[cur]['binstr'] <= 0:
                leaf_idx.append(cur) # 找到一个叶节点
                if self.skip_test(block_pntcnt[len(leaf_idx) - 1]):
                    blocks[cur]['binstr'] = -1 # 标记为跳过
                else:
                    blocks[cur]['binstr'] = 0 # 标记为转换
            else: # 划分当前节点
                idx = 0
                binstr = blocks[cur]['binstr']
                while binstr > 0:
                    if (binstr & 1) == 1: # 根据二进制字符串创建一个块
                        box = compute_new_bbox(idx, blocks[cur]['bbox_min'], blocks[cur]['bbox_max'])
                        block = {'level': blocks[cur]['level'] + 1, 'bbox_min': box[0], 'bbox_max': box[1]}
                        blocks.append(block)
                    idx += 1
                    binstr >>= 1
            cur += 1
            if cur >= len(blocks): break

        return [blocks[leaf_idx[i]] for i in range(len(leaf_idx))]


class OctreeOrganizer(OctreeConverter):
    """准备给定语法的八叉树数组和跳过块的数据，以便启用内部数据通信。"""

    def __init__(self, octree_cfg, max_num_points, syntax_gt, rw_octree=False, shuffle_blocks=False):

        # 获取八叉树划分的规格并创建一个八叉树转换器
        super().__init__(
            octree_cfg['bbox_min'],
            octree_cfg['bbox_max'],
            octree_cfg['point_min'],
            octree_cfg['point_max'],
            octree_cfg['level_min'],
            octree_cfg['level_max'],
        )

        # 设置八叉树划分选项
        self.syntax_gt = syntax_gt
        self.max_num_points = max_num_points
        self.rw_octree = rw_octree
        self.normalized_box_size = 2
        self.shuffle_blocks = shuffle_blocks
        self.infinitesimal = 1e-6

    def get_normalizer(self, bbox_min, bbox_max, pnts=None):
        center = (bbox_min + bbox_max) / 2
        scaling = self.normalized_box_size / (bbox_max[0] - bbox_min[0])
        return center, scaling


    def organize_data(self, points_raw, normal=None, file_name=None):
        if self.rw_octree and os.path.isfile(file_name): # 检查点云是否已经被转换为八叉树
            with open(file_name, 'rb') as f_pkl:
                octree_raw = pickle.load(f_pkl)
                blocks = octree_raw['blocks']
                leaf_idx = octree_raw['leaf_idx']
                points = octree_raw['points']
                binstrs = octree_raw['binstrs']
        else:
            # 执行八叉树划分
            blocks, leaf_idx, points, normal, binstrs = self.partition_octree(points_raw, normal)
            if self.rw_octree:
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                with open(file_name, "wb") as f_pkl: # 写下划分结果
                    pickle.dump({'blocks': blocks, 'leaf_idx': leaf_idx, 'points': points, 'normal': normal, 'binstrs': binstrs}, f_pkl)
    # 代码主要用于处理八叉树的去划分和组织。其中，departition_octree函数用于对给定的八叉树进行去划分，OctreeOrganizer类用于准备给定语法的八叉树数组和跳过块的数据，以便启用内部数据通信。
    
# 为批处理组织数据
        total_cnt = 0  # 初始化总计数为0
        points_out = np.zeros((self.max_num_points, self.syntax_gt['__len__']), dtype=np.float32)  # 初始化points_out为零矩阵
        normal_out = np.zeros((self.max_num_points, 3), dtype=np.float32) if normal is not None else None  # 如果normal不为空，初始化normal_out为零矩阵，否则为None
        block_pntcnt = []  # 初始化block_pntcnt为空列表

        # 只有在训练时才打乱块
        if self.shuffle_blocks: np.random.shuffle(leaf_idx)

        all_skip = True  # 初始化all_skip为True
        for idx in leaf_idx:  # 对于每一个idx
            pnt_start, pnt_end = blocks[idx]['pnt_range'][0], blocks[idx]['pnt_range'][1]  # 获取pnt_start和pnt_end
            xyz_slc = slice(pnt_start, pnt_end + 1)  # 创建切片
            cnt = pnt_end - pnt_start + 1  # 计算cnt

            # 如果我们还可以添加更多的块，那么继续
            if total_cnt + cnt <= self.max_num_points:
                block_slc = slice(total_cnt, total_cnt + cnt)  # 创建块切片
                center, scaling = self.get_normalizer(
                    blocks[idx]['bbox_min'], blocks[idx]['bbox_max'], points[xyz_slc, :])  # 获取中心和缩放
                points_out[block_slc, 0 : points.shape[1]] = points[xyz_slc, :] # x, y, z, 和其他存在的
                points_out[block_slc, self.syntax_gt['block_center'][0] : self.syntax_gt['block_center'][1] + 1] = center # 块的中心
                points_out[block_slc, self.syntax_gt['block_scale']] = scaling # 块的缩放
                points_out[block_slc, self.syntax_gt['block_pntcnt']] = cnt if (blocks[idx]['binstr'] >= 0) else -cnt # 块中的点数
                points_out[total_cnt, self.syntax_gt['block_start']] = 1 if (blocks[idx]['binstr'] >= 0) else -1 # 块的开始标志
                if normal is not None: normal_out[block_slc, :] = normal[xyz_slc, :]  # 如果normal不为空，更新normal_out
                if (blocks[idx]['binstr'] >= 0): all_skip = False  # 如果blocks[idx]['binstr']大于等于0，设置all_skip为False
                block_pntcnt.append(cnt)  # 将cnt添加到block_pntcnt
                total_cnt += cnt  # 更新total_cnt
            else: break  # 否则，跳出循环

        # 这里可以返回更多的东西，例如，关于跳过块的细节
        return points_out, normal_out, binstrs, np.asarray(block_pntcnt), all_skip

    # 这段代码主要用于为批处理组织数据。其中，total_cnt用于计数，points_out和normal_out用于存储输出的点和法线，block_pntcnt用于存储每个块中的点数，
    # all_skip用于标记是否所有的块都被跳过。如果你需要更多的帮助，比如解释这些函数的功能，或者你有其他的问题，欢迎随时向我提问。
