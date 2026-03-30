import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLoss(nn.Module):
    def __init__(self, lambd=0.1, beta=0.5, lambda_en=0.1, eps=1e-8):
        super(MyLoss, self).__init__()
        self.lambd = lambd
        self.beta = beta
        self.lambda_en = lambda_en
        self.eps = eps

    def forward(self, pred, truth, semantic, res_semantic, feature, deep_semantic, scene_probs=None):
        # 1. 分类项
        loss_cross_entropy = F.multilabel_soft_margin_loss(pred, truth)

        # 2. 语义对齐
        cosine_sim = torch.mean(torch.cosine_similarity(semantic, res_semantic, dim=1))

        # 3. 特征重建
        pred_feature = torch.matmul(pred, deep_semantic)
        loss_restructure = torch.norm(pred_feature - feature)

        # 4. 稀疏约束
        loss_sparse = torch.norm(pred)

        dsdl_loss = (
            loss_cross_entropy + self.beta * (loss_restructure + self.lambd * loss_sparse)
        ) / (cosine_sim + self.eps)

        # 5. entropy auxiliary loss
        if scene_probs is None:
            return dsdl_loss

        # sample-level entropy: 希望单样本scene分布更尖锐
        entropy_sample = -torch.mean(
            torch.sum(scene_probs * torch.log(scene_probs + self.eps), dim=1)
        )

        # batch-level entropy: 希望整个batch平均分布更均衡
        avg_scene_probs = torch.mean(scene_probs, dim=0)
        K = scene_probs.size(1)
        entropy_batch = torch.sum(avg_scene_probs * torch.log(avg_scene_probs + self.eps)) + torch.log(
            torch.tensor(float(K), device=scene_probs.device)
        )

        loss_en = entropy_sample + entropy_batch

        total_loss = dsdl_loss + self.lambda_en * loss_en
        return total_loss