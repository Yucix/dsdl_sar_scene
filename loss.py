import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLoss(nn.Module):
    def __init__(self, lambd=0.1, beta=0.5, eps=1e-8): # 这里的 beta 你可以后续再微调
        super(MyLoss, self).__init__()
        self.lambd = lambd
        self.beta = beta
        self.eps = eps

    def forward(self, pred, truth, semantic, res_semantic, feature, deep_semantic):
        # 1. 交叉熵
        loss_cross_entropy = F.multilabel_soft_margin_loss(pred, truth)
        
        # 2. 语义对齐 (修复为相似度)
        cosine_sim = torch.mean(torch.cosine_similarity(semantic, res_semantic, dim=1))
        
        # 3. 特征重建 (修复为官方写法，不除以 batch_size)
        pred_feature = torch.matmul(pred, deep_semantic)  
        loss_restructure = torch.norm(pred_feature - feature)
        
        # 4. 稀疏性约束 (修复为官方写法，不除以 numel)
        loss_sparse = torch.norm(pred)

        # 5. 总 Loss (相似度放分母)
        total_loss = (loss_cross_entropy + self.beta * (loss_restructure + self.lambd * loss_sparse)) / (cosine_sim + self.eps)

        return total_loss