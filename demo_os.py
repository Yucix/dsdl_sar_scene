import argparse
import os
import sys
import torch
import torch.optim
import csv
import datetime
import numpy as np
import random  

# 保证 src 模块能被正确导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import DSDLMultiLabelMAPEngine
from os_dataset import OSDataset
from loss import MyLoss
from models import load_model


# ===============================
# 默认路径
# ===============================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_PATH = "/media/sata/xyx/dsdl/dataset"
DEFAULT_EMBEDDING_PATH = os.path.join(DEFAULT_DATA_PATH, "embeddings/s2_glove_word2vec.pkl")
DEFAULT_CHECKPOINT_PATH = "/media/sata/xyx/dsdl/checkpoints/dsdl_sar/"
DEFAULT_LOG_PATH = os.path.join(project_root, "/media/sata/xyx/dsdl/logs/dsdl_sar/")

# 保证实验可复现的随机种子
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(3407)

# =================================
# 日志记录类
# =================================
class TrainingLogger:
    def __init__(self, log_dir=DEFAULT_LOG_PATH):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"training_log_{timestamp}.csv")
        self.init_csv()

    def init_csv(self):
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 新增列：Macro_P, Macro_R, Macro_F1
            writer.writerow([
                'timestamp', 'epoch', 'phase', 'loss',
                'backbone_lr', 'semantic_lr',
                'mAP', 'Macro_P', 'Macro_R', 'Macro_F1', 'Micro_F1',
                'AP_per_class', 'F1_per_class', 'epoch_time'
            ])

    def log_epoch(self, epoch, phase, loss, lr, metrics, epoch_time):
        # ... (lr 处理逻辑不变) ...
        if isinstance(lr, (list, tuple)):
            backbone_lr, semantic_lr = lr[0], lr[-1]
        elif hasattr(lr, "__len__"):
            backbone_lr, semantic_lr = lr[0], lr[-1]
        elif lr is None:
            backbone_lr = semantic_lr = ''
        else:
            backbone_lr = semantic_lr = lr

        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                epoch,
                phase,
                f"{loss:.6f}" if loss is not None else '',
                f"{float(backbone_lr):.8f}" if backbone_lr != '' else '',
                f"{float(semantic_lr):.8f}" if semantic_lr != '' else '',
                metrics.get('mAP', ''),
                metrics.get('Macro_P', ''),
                metrics.get('Macro_R', ''),
                metrics.get('Macro_F1', ''),
                metrics.get('Micro_F1', ''),
                metrics.get('AP_per_class', ''),
                metrics.get('F1_per_class', ''),
                f"{epoch_time:.3f}"
            ])

    def log_best_model(self, best_metrics):
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["==== Best Model Summary (Based on Micro-F1) ===="])
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "BEST_MODEL",
                "", "", "", "",
                best_metrics.get('mAP', ''),
                best_metrics.get('Macro_P', ''),
                best_metrics.get('Macro_R', ''),
                best_metrics.get('Macro_F1', ''),
                best_metrics.get('Micro_F1', ''),
                best_metrics.get('AP_per_class', ''),
                best_metrics.get('F1_per_class', ''),
                ""
            ])


# ===============================
# 参数解析
# ===============================
parser = argparse.ArgumentParser(description='OS Dataset Training (Optical + SAR Fusion)')
parser.add_argument('--data', default=DEFAULT_DATA_PATH, type=str)
parser.add_argument('--image-size', '-i', default=256, type=int)
parser.add_argument('--device_ids', default=[0], type=int, nargs='+')
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--epoch_step', default=[30, 60], type=int, nargs='+')
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=32, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lrp', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('-p', '--print-freq', default=0, type=int)
parser.add_argument('--resume', default='', type=str, help='path to checkpoint to resume from')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--lambd', default=0.001, type=float)
parser.add_argument('--beta', default=0.005, type=float)
parser.add_argument('--log-dir', default=DEFAULT_LOG_PATH, type=str)
parser.add_argument('--early-stop', action='store_true',
                    help='enable early stopping based on validation Micro-F1')
parser.add_argument('--patience', default=15, type=int,
                    help='number of epochs with no improvement before stopping')
parser.add_argument('--sar-patch-size', default=16, type=int)
parser.add_argument('--sar-embed-dim', default=64, type=int)
parser.add_argument('--sar-num-vig-blocks', default=2, type=int)
parser.add_argument('--sar-num-segments', default=64, type=int)
parser.add_argument('--sar-num-edges', default=9, type=int)
parser.add_argument('--sar-head-num', default=1, type=int)
parser.add_argument('--sar-drop-path', default=0.05, type=float)

# ===============================
# main function
# ===============================
def main_os():
    args = parser.parse_args()

    global logger
    logger = TrainingLogger(args.log_dir)

    print("############################################")
    print(" Optical + SAR Fusion DSDL Training ")
    print("############################################")
    print(f"Data path: {args.data}")
    print(f"Log path:  {logger.log_dir}")
    if args.resume:
        print(f"Resume from checkpoint: {args.resume}")

    # ============ Dataset ============
    train_dataset = OSDataset(
        root=args.data,
        set="train",
        transform=None,
        inp_name=DEFAULT_EMBEDDING_PATH,
        num_segments=args.sar_num_segments,
        patch_size=args.sar_patch_size,
    )
    val_dataset = OSDataset(
        root=args.data,
        set="test",
        transform=None,
        inp_name=DEFAULT_EMBEDDING_PATH,
        num_segments=args.sar_num_segments,
        patch_size=args.sar_patch_size,
    )

    # ============ Model ============
    num_classes = 6

    model = load_model(
    num_classes=num_classes,
    alpha=args.lambd,
    sar_patch_size=args.sar_patch_size,
    sar_embed_dim=args.sar_embed_dim,
    sar_num_vig_blocks=args.sar_num_vig_blocks,
    sar_num_segments=args.sar_num_segments,
    sar_num_edges=args.sar_num_edges,
    sar_head_num=args.sar_head_num,
    sar_drop_path=args.sar_drop_path,
)

    # ============ Loss & Optimizer ============
    criterion = MyLoss(args.lambd, args.beta)
    #  AdamW 优化器
    optimizer = torch.optim.AdamW(
        model.get_config_optim(args.lr, args.lrp),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # ============ Engine ============
    state = {
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'max_epochs': args.epochs,
        'evaluate': args.evaluate,
        'resume': args.resume,
        'num_classes': num_classes,
        'workers': args.workers,
        'epoch_step': args.epoch_step,
        'lr': args.lr,
        'device_ids': args.device_ids,
        'dataset': 'os',
        'logger': logger,
        'save_model_path': DEFAULT_CHECKPOINT_PATH,
        'early_stop': args.early_stop,
        'patience': args.patience,
    }

    engine = DSDLMultiLabelMAPEngine(state)
    best_score = engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

    # 打印和记录最佳结果
    best_metrics = getattr(engine, 'best_metrics', {})
    if not best_metrics:
        best_metrics = {'Micro_F1': f"{best_score:.4f}"}

    logger.log_best_model(best_metrics)

    print("############################################")
    print(" Training Complete! ")
    print("############################################")
    print(f"Best Micro-F1 = {best_metrics.get('Micro_F1', best_score)}")
    print(f"Best Epoch    = {engine.state.get('best_epoch', 'N/A')}")
    print(f"Macro-F1      = {best_metrics.get('Macro_F1', 'N/A')}")
    print(f"Macro-P       = {best_metrics.get('Macro_P', 'N/A')}")
    print(f"Macro-R       = {best_metrics.get('Macro_R', 'N/A')}")
    print("--------------------------------------------")
    print(f"Best model saved in: {DEFAULT_CHECKPOINT_PATH}")
    print(f"Log file saved in:   {logger.log_file}")
    print("############################################")

if __name__ == "__main__":
    main_os()