import os
import json
import torch
import torch.utils.data as data
import numpy as np
import rasterio
import random
import torchvision.transforms.functional as TF

s2_categories = ['water', 'road', 'building',
                 'barren', 'vegetation', 'farmland']

class OSDataset(data.Dataset):
    def __init__(self, root, set='train', transform=None, target_transform=None, inp_name=None, num_segments=64, patch_size=8):
        self.root = root
        self.set = set
        self.num_segments = num_segments
        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = target_transform

        self.optical_dir = os.path.join(root, set, "optical")
        self.sar_dir = os.path.join(root, set, "sar")
        self.nodes_dir = os.path.join(root, set, f"aug_nodes_slico_seg{num_segments}_patch{patch_size}")
        self.label_path = os.path.join(root, set, "labels.json")

        with open(self.label_path, "r") as f:
            self.labels = json.load(f)

        self.files = list(self.labels.keys())
        self.num_classes = len(s2_categories)

        if inp_name and os.path.exists(inp_name):
            import pickle
            with open(inp_name, 'rb') as f:
                self.inp = torch.tensor(pickle.load(f), dtype=torch.float32)
        else:
            self.inp = torch.randn(self.num_classes, 300)

        print(f"[OSDataset] {set}: {len(self.files)} samples loaded.")

    def load_tiff(self, path):
        with rasterio.open(path) as src:
            img = src.read()
            img = np.transpose(img, (1, 2, 0))
        return img

    def normalize_optical(self, opt):
        h, w, c = opt.shape
        if c >= 4:
            opt = opt[:, :, :4]
        else:
            pad = np.zeros((h, w, 4 - c), dtype=opt.dtype)
            opt = np.concatenate([opt, pad], axis=2)
        return opt

    def normalize_sar(self, sar):
        if sar.ndim == 2:
            sar = sar[:, :, None]
        return sar[:, :, 0:1]

    def __getitem__(self, index):
        fname_opt = self.files[index]
        opt_path = os.path.join(self.optical_dir, fname_opt)
        sar_name = fname_opt.replace("S2", "S1")
        sar_path = os.path.join(self.sar_dir, sar_name)

        opt = self.load_tiff(opt_path)
        opt = self.normalize_optical(opt)
        sar = self.load_tiff(sar_path)
        sar = self.normalize_sar(sar)

        fusion = np.concatenate([opt, sar], axis=2)
        fusion = torch.from_numpy(fusion).permute(2, 0, 1).float()  # [5,H,W]

        fusion = TF.resize(fusion, [256, 256], antialias=True)

        # 核心：离散增强同步逻辑
        aug_type = "orig" # 默认或测试集使用原图
        
        if self.set == "train":
            aug_type = random.choice(["orig", "hflip", "vflip", "rot180"])
            
            # 对图像 Tensor 进行绝对同步的空间变换
            if aug_type == "hflip":
                fusion = TF.hflip(fusion)
            elif aug_type == "vflip":
                fusion = TF.vflip(fusion)
            elif aug_type == "rot180":
                fusion = torch.rot90(fusion, k=2, dims=[1, 2])

        # 读取与其空间状态完全对应的预存 nodes
        node_name = sar_name.replace(".tif", f"_{aug_type}.npy")
        node_path = os.path.join(self.nodes_dir, node_name)
        
        if not os.path.exists(node_path):
            raise FileNotFoundError(f"Missing precomputed node file: {node_path}")
            
        nodes = np.load(node_path)
        nodes = torch.from_numpy(nodes).float()

        # 标签提取
        labels = self.labels[fname_opt]
        target = torch.zeros(self.num_classes)
        for lb in labels:
            if lb in s2_categories:
                target[s2_categories.index(lb)] = 1

        if self.target_transform:
            target = self.target_transform(target)

        return (fusion, fname_opt, [self.inp], nodes), target

    def __len__(self):
        return len(self.files)