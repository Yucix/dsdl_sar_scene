import os
import json
import torch
import torch.utils.data as data
import numpy as np
import rasterio

# 类别
s2_categories = ['water', 'road', 'building',
                 'barren', 'vegetation', 'farmland']


class OSDataset(data.Dataset):
    """
    Optical(S2) + SAR(S1) 融合数据集：
    - optical: root/set/optical/*.tif
    - sar:     root/set/sar/*.tif
    - labels:  root/set/labels.json, key = optical 文件名，value = 标签列表
    输出:
    - fusion: Tensor [5, H, W]  (4 optical + 1 SAR)
    - name:   光学图像文件名
    - [inp]:  类别词向量 [num_classes, 300]
    - target: 多标签 one-hot [num_classes]
    """

    def __init__(self, root, set='train', transform=None, target_transform=None, inp_name=None):
        self.root = root
        self.set = set

        self.optical_dir = os.path.join(root, set, "optical")
        self.sar_dir = os.path.join(root, set, "sar")
        self.label_path = os.path.join(root, set, "labels.json")

        self.transform = transform
        self.target_transform = target_transform

        # 加载标签
        with open(self.label_path, "r") as f:
            self.labels = json.load(f)

        self.files = list(self.labels.keys())
        self.num_classes = len(s2_categories)

        # 加载词向量（类别语义向量）
        if inp_name and os.path.exists(inp_name):
            import pickle
            with open(inp_name, 'rb') as f:
                self.inp = torch.tensor(pickle.load(f), dtype=torch.float32)
        else:
            self.inp = torch.randn(self.num_classes, 300)

        print(f"[OSDataset] {set}: {len(self.files)} samples loaded.")

    # 读取 tiff 文件 → numpy (H, W, C)
    def load_tiff(self, path):
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))  # → (H, W, C)
        return img

    # 固定 Optical 为 4 通道（不足补零，超过截取前4）
    def normalize_optical(self, opt):
        h, w, c = opt.shape
        if c >= 4:
            opt = opt[:, :, :4]
        else:
            pad = np.zeros((h, w, 4 - c), dtype=opt.dtype)
            opt = np.concatenate([opt, pad], axis=2)
        return opt  # (H, W, 4)

    # 统一 SAR 为 1 通道（取第一个通道）
    def normalize_sar(self, sar):
        if sar.ndim == 2:
            sar = sar[:, :, None]
        return sar[:, :, 0:1]  # (H, W, 1)

    def __getitem__(self, index):
        fname_opt = self.files[index]

        # 路径
        opt_path = os.path.join(self.optical_dir, fname_opt)
        sar_name = fname_opt.replace("S2", "S1")
        sar_path = os.path.join(self.sar_dir, sar_name)

        # ==========================================
        # 1. 光学数据 
        # ==========================================
        opt = self.load_tiff(opt_path)
        opt = self.normalize_optical(opt)

        # ==========================================
        # 2. 雷达数据 
        # ==========================================
        sar = self.load_tiff(sar_path)
        sar = self.normalize_sar(sar)

        # ==========================================
        # 3. 拼接 (不变)
        # ==========================================
        fusion = np.concatenate([opt, sar], axis=2)

        # # ==========================================
        # # 4. 标准化 (标准化参数调整)
                
        # # 对标dsdl代码中engine的normalize部分
        # mean = [0.485, 0.456, 0.406, 0.5, 0.5] 
        # std  = [0.229, 0.224, 0.225, 0.5, 0.5]
        
        # for c in range(5):
        #     fusion[:, :, c] = (fusion[:, :, c] - mean[c]) / std[c]

        # 转 tensor: (5, H, W)
        fusion = torch.from_numpy(fusion).permute(2, 0, 1)

        # Tensor 级 transform（由 engine 注入）
        if self.transform:
            fusion = self.transform(fusion)

        # 标签 → one-hot
        labels = self.labels[fname_opt]
        target = torch.zeros(self.num_classes)
        for lb in labels:
            if lb in s2_categories:
                target[s2_categories.index(lb)] = 1

        if self.target_transform:
            target = self.target_transform(target)

        return (fusion, fname_opt, [self.inp]), target

    def __len__(self):
        return len(self.files)
