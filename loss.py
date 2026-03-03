import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLoss(nn.Module):
    def __init__(self, lambd=0.1, beta=0.5, gamma=1, eps=1e-8): # 新增 gamma 控制熵损失权重
        super(MyLoss, self).__init__()
        self.lambd = lambd
        self.beta = beta
        self.eps = eps  # 防止除0
        self.gamma = gamma

    def forward(self, pred, truth, semantic, res_semantic, feature, deep_semantic, scene_prob):
        # 1. DSDL 基础 Loss
        loss_cross_entropy = F.multilabel_soft_margin_loss(pred, truth)
        
        cosine_sim = torch.mean(torch.cosine_similarity(semantic, res_semantic, dim=1))
        
        # feature是2048维
        pred_feature = torch.matmul(pred, deep_semantic)  
        loss_restructure = torch.norm(pred_feature - feature)
        loss_sparse = torch.norm(pred)

        # 分子 / 相似度 (由于相似度目标是趋近1，放分母既能优化又不会爆炸，加上eps防初始极少数异常)
        base_loss = (loss_cross_entropy + self.beta * (loss_restructure + self.lambd * loss_sparse)) / (cosine_sim + self.eps)

        # 2. 场景分布熵损失 (防止赢者通吃)
        K = scene_prob.size(1)
        # 样本级熵 (越小越好)
        loss_sample_entropy = -torch.mean(torch.sum(scene_prob * torch.log(scene_prob + self.eps), dim=1))
        # 批次级熵 (越小越好)
        avg_prob = torch.mean(scene_prob, dim=0)
        import math
        loss_batch_entropy = torch.sum(avg_prob * torch.log(avg_prob + self.eps)) - math.log(1.0 / K)
        
        loss_entropy = loss_sample_entropy + loss_batch_entropy

        # 3. 总 Loss
        total_loss = base_loss + self.gamma * loss_entropy

        return total_loss