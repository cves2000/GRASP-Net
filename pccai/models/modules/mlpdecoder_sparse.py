def prepare_meta_data(self, binstrs, block_pntcnt, octree_organizer):
        """将八叉树的二进制字符串转换为叶节点的一组比例和中心，然后根据解码的语法将它们组织为元数据数组。"""

        # 将八叉树字符串分区为块
        leaf_blocks = octree_organizer.departition_octree(binstrs, block_pntcnt)
        # 初始化元数据数组
        meta_data = np.zeros((len(leaf_blocks), self.syntax_cw['__len__'] - self.syntax_cw['__meta_idx__']), dtype=np.float32)
        cur = 0

        # 组装元数据
        meta_data[0, self.syntax_cw['pc_start'] - self.syntax_cw['__meta_idx__']] = 1
        for idx, block in enumerate(leaf_blocks):
            if block['binstr'] >= 0: # 只保留具有变换模式的块
                # 获取块的中心和规模
                center, scale = octree_organizer.get_normalizer(block['bbox_min'], block['bbox_max'])
                # 将块的点数、规模和中心添加到元数据中
                meta_data[cur, self.syntax_cw['block_pntcnt'] - self.syntax_cw['__meta_idx__']] = block_pntcnt[idx]
                meta_data[cur, self.syntax_cw['block_scale'] - self.syntax_cw['__meta_idx__']] = scale
                meta_data[cur, self.syntax_cw['block_center'][0] - self.syntax_cw['__meta_idx__'] : 
                    self.syntax_cw['block_center'][1] - self.syntax_cw['__meta_idx__'] + 1] = center
                cur += 1

        # 只返回有用的部分，并将其转换为PyTorch张量
        return torch.as_tensor(meta_data[:cur, :], device=torch.device('cuda')).unsqueeze(-1).unsqueeze(-1)
