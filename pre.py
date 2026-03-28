import os
import numpy as np
import rasterio
import cv2
from skimage.segmentation import slic
from skimage.util import img_as_float
from tqdm import tqdm

def resize_patch_np(patch, patch_size):
    if patch.shape[0] == 0 or patch.shape[1] == 0:
        return np.zeros((patch_size, patch_size), dtype=np.float32)

    patch = patch.astype(np.float32)
    patch_resized = cv2.resize(
        patch,
        (patch_size, patch_size),
        interpolation=cv2.INTER_LINEAR
    )
    return patch_resized

def build_nodes_from_labels(sar_img, labels, patch_size=8):
    nodes = []
    num_sp = int(labels.max()) + 1

    for seg_id in range(num_sp):
        mask = (labels == seg_id)
        if not np.any(mask):
            continue

        ys, xs = np.where(mask)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1

        patch = sar_img[y1:y2, x1:x2].copy()
        patch_mask = mask[y1:y2, x1:x2].astype(np.float32)
        patch = patch * patch_mask

        patch_resized = resize_patch_np(patch, patch_size)
        nodes.append(patch_resized.reshape(-1))

    if len(nodes) == 0:
        nodes = np.zeros((1, patch_size * patch_size), dtype=np.float32)
    else:
        nodes = np.stack(nodes, axis=0).astype(np.float32)

    return nodes

def process_and_save(sar_img, num_segments, patch_size, out_path):
    """核心处理逻辑：算超像素 -> 提节点 -> 保存"""
    labels = slic(
        img_as_float(sar_img),
        n_segments=num_segments,
        slic_zero=True,
        start_label=0,
        channel_axis=None
    ).astype(np.int16)

    nodes = build_nodes_from_labels(sar_img=sar_img, labels=labels, patch_size=patch_size)
    np.save(out_path, nodes)

def precompute_slico_nodes(data_root, num_segments=64, patch_size=8):
    for split in ["train", "test"]:
        sar_dir = os.path.join(data_root, split, "sar")
        out_dir = os.path.join(data_root, split, f"aug_nodes_slico_seg{num_segments}_patch{patch_size}")
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(sar_dir):
            continue

        files = os.listdir(sar_dir)
        print(f"==============")
        print(f"处理 {split} 集，保存至 {out_dir}")

        for f in tqdm(files, desc=f"{split} 进度"):
            if not f.endswith(".tif"):
                continue

            sar_path = os.path.join(sar_dir, f)
            with rasterio.open(sar_path) as src:
                img = src.read()
                img = np.transpose(img, (1, 2, 0))

            sar_img = img[:, :, 0]  # [H, W]

            # 1. 始终生成并保存原图
            process_and_save(sar_img, num_segments, patch_size, os.path.join(out_dir, f.replace(".tif", "_orig.npy")))

            # 2. 如果是训练集，额外生成3种几何变换版本
            if split == "train":
                # 水平翻转
                process_and_save(np.fliplr(sar_img), num_segments, patch_size, os.path.join(out_dir, f.replace(".tif", "_hflip.npy")))
                # 垂直翻转
                process_and_save(np.flipud(sar_img), num_segments, patch_size, os.path.join(out_dir, f.replace(".tif", "_vflip.npy")))
                # 旋转180度
                process_and_save(np.rot90(sar_img, k=2), num_segments, patch_size, os.path.join(out_dir, f.replace(".tif", "_rot180.npy")))

if __name__ == "__main__":
    DATA_ROOT = "/media/sata/xyx/dsdl/dataset"
    precompute_slico_nodes(DATA_ROOT, num_segments=64, patch_size=16)
    print("全部多视角 nodes 预计算完成！")