import os
import shutil
import time
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchnet as tnt
import numpy as np
from tqdm import tqdm
import math

from util import AveragePrecisionMeter



def os_node_collate_fn(batch):
    """
    batch: list of [ ((fusion, fname, [inp], nodes), target), ... ]
    返回：
        (fusion_batch, fname_list, [inp], nodes_batch, node_mask), target_batch
    """
    inputs, targets = zip(*batch)

    fusions = []
    fnames = []
    inps = []
    nodes_list = []

    for item in inputs:
        fusion, fname, inp, nodes = item
        fusions.append(fusion)
        fnames.append(fname)
        inps.append(inp[0])   # 原来 inp 是 [self.inp]
        nodes_list.append(nodes)

    fusion_batch = torch.stack(fusions, dim=0)
    inp_batch = torch.stack(inps, dim=0)
    target_batch = torch.stack(targets, dim=0)

    max_nodes = max(n.shape[0] for n in nodes_list)
    feat_dim = nodes_list[0].shape[1]

    padded_nodes = []
    node_masks = []

    for nodes in nodes_list:
        n = nodes.shape[0]
        pad_n = max_nodes - n

        if pad_n > 0:
            padding = torch.zeros(pad_n, feat_dim, dtype=nodes.dtype)
            nodes = torch.cat([nodes, padding], dim=0)

        mask = torch.zeros(max_nodes, dtype=torch.bool)
        mask[:n] = True

        padded_nodes.append(nodes)
        node_masks.append(mask)

    nodes_batch = torch.stack(padded_nodes, dim=0)   # [B, Nmax, F]
    node_mask = torch.stack(node_masks, dim=0)       # [B, Nmax]

    return (fusion_batch, fnames, [inp_batch], nodes_batch, node_mask), target_batch

class Engine(object):
    def __init__(self, state=None):
        if state is None:
            state = {}
        self.state = state
        self.best_metrics = None

        self.state.setdefault('use_gpu', torch.cuda.is_available())
        self.state.setdefault('image_size', 256)
        self.state.setdefault('batch_size', 32)
        self.state.setdefault('workers', 4)
        self.state.setdefault('device_ids', None)
        self.state.setdefault('evaluate', False)
        self.state.setdefault('start_epoch', 0)
        self.state.setdefault('max_epochs', 100)
        self.state.setdefault('epoch_step', [])
        self.state.setdefault('save_model_path', './checkpoints/')
        self.state.setdefault('best_score', 0)
        self.state.setdefault('early_stop', False)
        self.state.setdefault('patience', 15)
        self.state.setdefault('best_epoch', -1)
        self.state.setdefault('epochs_no_improve', 0)


        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()

    # ===== epoch 级别 =====
    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()
        if not training:
            self.state['meter_ap'] = AveragePrecisionMeter()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None):
        epoch_time = time.time() - self.state.get('epoch_start_time', time.time())
        loss = self.state['meter_loss'].value()[0]

        print(f"{'Train' if training else 'Val'} Epoch [{self.state['epoch']}]: Loss {loss:.4f}")

        metrics_log = {}
        score_to_return = 0.0

        # 验证阶段：计算指标
        if not training:
            ap_meter = self.state['meter_ap']
            
            res = ap_meter.compute_paper_metrics(threshold=0.5)
            
            # 核心指标
            macro_p = res['Macro_P']
            macro_r = res['Macro_R']
            macro_f1 = res['Macro_F1']
            micro_f1 = res['Micro_F1']
            mAP = res['mAP']

            ap_per_class = res['Per_Class_AP']
            f1_per_class = res['Per_Class_F1']

            print("\n" + "="*45)
            print(" *** Evaluation Results (Aligned with Paper) ***")
            print("="*45)
            print(f" Micro-F1 (Target): {micro_f1:.2f} %")
            print(f" Macro-F1 (Paper F1): {macro_f1:.2f} %")
            print(f" Macro-P (Paper P)  : {macro_p:.2f} %")
            print(f" Macro-R (Paper R)  : {macro_r:.2f} %")
            print(f" mAP              : {mAP:.4f}")
            print("-" * 45)
            print(f" Per-class F1     : {np.round(f1_per_class, 2)}")
            print("="*45 + "\n")

            # === 准备日志 ===
            metrics_log = {
                "mAP": f"{mAP:.6f}",
                "Macro_P": f"{macro_p:.4f}",
                "Macro_R": f"{macro_r:.4f}",
                "Macro_F1": f"{macro_f1:.4f}",
                "Micro_F1": f"{micro_f1:.4f}",
                "AP_per_class": ",".join([f"{x:.4f}" for x in ap_per_class]),
                "F1_per_class": ",".join([f"{x:.2f}" for x in f1_per_class]),
            }
            
            # 使用 Micro-F1 作为最优模型的判断标准
            score_to_return = micro_f1

        # 写日志
        logger = self.state.get('logger', None)
        if logger is not None:
            lr_value = None
            if 'lr' in self.state:
                lr_state = self.state['lr']
                lr_value = lr_state[0] if isinstance(lr_state, (list, tuple)) else lr_state

            logger.log_epoch(
                epoch=self.state['epoch'],
                phase='train' if training else 'val',
                loss=loss,
                lr=lr_value,
                metrics=metrics_log,
                epoch_time=epoch_time
            )

        return score_to_return if not training else loss

    # ===== batch 级别 =====
    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None):
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch'])

    def on_forward(self, training, model, criterion, data_loader, optimizer=None):
        pass

    # ===== 训练配置 =====
    def init_learning(self, model, criterion):
        self.state['train_transform'] = None
        self.state['val_transform'] = None
        
    def adjust_learning_rate(self, optimizer):
        epoch = self.state['epoch']
        max_epochs = self.state['max_epochs']
        warmup_epochs = 5  # 设置前 5 个 epoch 为 Warmup 阶段
        
        # 第一次调用时，记录每个参数组的初始学习率
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']
                
        # 计算当前 epoch 的学习率缩放比例 (scale)
        if epoch < warmup_epochs:
            # 线性 Warmup：从很小的值线性增长到 1.0
            scale = (epoch + 1) / warmup_epochs
        else:
            # 余弦退火 (Cosine Annealing)
            # 将剩余的 epoch 映射到 0 ~ pi 区间
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # 设定一个最小学习率底线 (例如初始学习率的 1%)
            min_scale = 0.01 
            scale = min_scale + (1.0 - min_scale) * scale

        # 应用算出的比例
        lr_list = []
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * scale
            lr_list.append(param_group['lr'])
            
        return np.unique(lr_list)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        path = self.state['save_model_path']
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(
                filepath,
                os.path.join(path, f"model_best_{state['best_score']:.4f}.pth.tar")
            )
            
    # ===== 训练流程 =====
    def learning(self, model, criterion, train_dataset, val_dataset, optimizer):
        self.init_learning(model, criterion)
        train_dataset.transform = self.state['train_transform']
        val_dataset.transform = self.state['val_transform']
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.state['batch_size'],
            shuffle=True,
            num_workers=self.state['workers'],
            pin_memory=True,
            persistent_workers=True if self.state['workers'] > 0 else False,
            collate_fn=os_node_collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.state['batch_size'],
            shuffle=False,
            num_workers=self.state['workers'],
            pin_memory=True,
            persistent_workers=True if self.state['workers'] > 0 else False,
            collate_fn=os_node_collate_fn,
        )

        if self.state['use_gpu']:
            model = nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
            criterion = criterion.cuda()

        # Resume 断点续训逻辑
        # ==========================================
        if self.state.get('resume'):
            if os.path.isfile(self.state['resume']):
                print(f"=> Loading checkpoint '{self.state['resume']}'")
                checkpoint = torch.load(self.state['resume'])
                
                # 1. 恢复起始 epoch 和最高分
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                
                # 2. 恢复模型权重
                if self.state['use_gpu']:
                    model.module.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint['state_dict'])
                
                # 3. 恢复优化器状态 
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> Optimizer state loaded successfully.")
                else:
                    print("=> Warning: No optimizer state found in checkpoint. Starting optimizer from scratch.")
                    
                print(f"=> Loaded checkpoint (Epoch {checkpoint['epoch']}, Best Score: {checkpoint['best_score']:.4f})")
            else:
                print(f"=> No checkpoint found at '{self.state['resume']}'")
        # ==========================================

        if self.state['evaluate']:
            return self.validate(val_loader, model, criterion)

        patience = self.state.get('patience', 15)
        epochs_no_improve = 0

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            self.state['lr'] = lr
            print(f"Learning rate: {lr}")

            self.train(train_loader, model, criterion, optimizer)
            score = self.validate(val_loader, model, criterion)

            is_best = (score is not None) and (score > self.state['best_score'])

            if is_best:
                self.state['best_score'] = score
                self.state['best_epoch'] = epoch + 1
                self.best_metrics = self.state['meter_ap'].compute_paper_metrics()
                epochs_no_improve = 0
                print(f"=> New best Micro-F1: {score:.4f} at epoch {epoch + 1}")
            else:
                epochs_no_improve += 1
                print(f"=> No improvement for {epochs_no_improve} epoch(s). "
                    f"Best Micro-F1: {self.state['best_score']:.4f} at epoch {self.state.get('best_epoch', -1)}")

            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
                'best_epoch': self.state.get('best_epoch', -1),
                'optimizer': optimizer.state_dict()
            }, is_best)

            if self.state.get('early_stop', False) and epochs_no_improve >= patience:
                print(f"=> Early stopping triggered at epoch {epoch + 1}. "
                    f"Best Micro-F1: {self.state['best_score']:.4f} at epoch {self.state.get('best_epoch', -1)}")
                break
        return self.state['best_score']
    
    def train(self, data_loader, model, criterion, optimizer):
        self.state['epoch_start_time'] = time.time()
        model.train()
        self.on_start_epoch(True, model, criterion, data_loader, optimizer)
        
        for i, (input, target) in enumerate(tqdm(data_loader, desc='Training')):
            self.state['iteration'] = i
            self.state['input'] = input
            self.state['target'] = target
            self.on_start_batch(True, model, criterion, data_loader, optimizer)
            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(non_blocking=True)
            self.on_forward(True, model, criterion, data_loader, optimizer)
            self.on_end_batch(True, model, criterion, data_loader, optimizer)
            
        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):
        self.state['epoch_start_time'] = time.time()
        model.eval()
        self.on_start_epoch(False, model, criterion, data_loader)
        
        with torch.no_grad():
            for i, (input, target) in enumerate(tqdm(data_loader, desc='Validation')):
                self.state['iteration'] = i
                self.state['input'] = input
                self.state['target'] = target
                self.on_start_batch(False, model, criterion, data_loader)
                if self.state['use_gpu']:
                    self.state['target'] = self.state['target'].cuda(non_blocking=True)
                self.on_forward(False, model, criterion, data_loader)
                self.on_end_batch(False, model, criterion, data_loader)
                
        return self.on_end_epoch(False, model, criterion, data_loader)


class DSDLMultiLabelMAPEngine(Engine):
    """ DSDL Multi-label Engine """
    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None):
        self.state['target_gt'] = self.state['target'].clone()
        input = self.state['input']
        fusion = input[0] # [B, 5, H, W]
        
        self.state['opt'] = fusion[:, :4, :, :]
        self.state['sar'] = fusion[:, 4:5, :, :]
        self.state['out'] = input[1] # filenames
        
        inp = input[2]
        if isinstance(inp, (list, tuple)):
            inp = inp[0]
        self.state['input'] = inp # semantic vectors

        self.state['sar_nodes'] = input[3]   # [B, N, F]
        self.state['node_mask'] = input[4]   # [B, N]

    def on_forward(self, training, model, criterion, data_loader, optimizer=None):
        opt_var = self.state['opt'].float()
        sar_var = self.state['sar'].float()
        target_var = self.state['target'].float()
        inp_var = self.state['input'].float()

        sar_nodes = self.state['sar_nodes'].float()
        node_mask = self.state['node_mask']

        if self.state['use_gpu']:
            sar_nodes = sar_nodes.cuda(non_blocking=True)
            node_mask = node_mask.cuda(non_blocking=True)
            opt_var = opt_var.cuda(non_blocking=True)
            sar_var = sar_var.cuda(non_blocking=True)
            inp_var = inp_var.cuda(non_blocking=True)

        scene_warmup = self.state.get('scene_warmup', 5)
        update_scene_counts = training and (self.state['epoch'] >= scene_warmup)

        self.state['output'], semantic, res_semantic, feature, deep_semantic, scene_probs = model(
            opt_var,
            sar_var,
            inp_var,
            sar_nodes,
            node_mask,
            target=target_var,
            update_scene_counts=update_scene_counts
        )

        self.state['loss'] = criterion(
            self.state['output'],
            target_var,
            semantic,
            res_semantic,
            feature,
            deep_semantic,
            scene_probs
        )

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None):
        super().on_end_batch(training, model, criterion, data_loader, optimizer)
        if not training:
            if 'meter_ap' not in self.state:
                self.state['meter_ap'] = AveragePrecisionMeter()
            self.state['meter_ap'].add(self.state['output'].detach(), self.state['target_gt'])