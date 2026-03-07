import math
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from vig import ViGBlock
from sg import SimplePatchifier, BN_Layer

def modify_resnet_conv1(model: nn.Module, in_channels: int) -> nn.Module:
    """Modify ResNet conv1 to accept arbitrary input channels (here: 4 for optical, 1 for SAR).

    - For 4ch optical: copy pretrained RGB weights to first 3 channels, duplicate red to 4th channel.
    - For 1ch SAR: average pretrained RGB weights across channel dimension.
    """
    old_conv = model.conv1
    old_w = old_conv.weight.data.clone()  # (64, 3, 7, 7)

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        if in_channels == 4:
            new_conv.weight[:, :3, :, :].copy_(old_w)
            new_conv.weight[:, 3:4, :, :].copy_(old_w[:, 2:3, :, :])  # copy red
        elif in_channels == 1:
            new_conv.weight.copy_(old_w.mean(dim=1, keepdim=True))
        else:
            raise ValueError(f"Unsupported in_channels={in_channels}. Expected 1 or 4.")

    model.conv1 = new_conv
    return model

    
class SARViGBackbone(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, embed_dim=320, num_blocks=8, num_patches=256, out_dim=2048,drop_path=0.1):
        super().__init__()
        
        # 1. 选点与切片 (Patchifier)
        # 将 256x256 的图切成 256 个 patch 
        self.patchifier = SimplePatchifier(patch_size=patch_size, num_patches=num_patches)
        
        # 2. Patch Embedding
        # 将每个 patch 映射为 embed_dim 
        self.patch_embedding = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        self.pose_embedding = nn.Parameter(torch.Tensor(1, num_patches, embed_dim))
        
        # 3. ViG Backbone (深层图网络)
        # 堆叠多个 ViG Block 来替代 ResNet 的层
        # num_blocks 可以设为 6, 8, 12 等，层数越深特征越强
        self.blocks = nn.Sequential(*[
            ViGBlock(in_features=embed_dim, num_edges=9, head_num=4, drop_path=drop_path) 
            for _ in range(num_blocks)
        ])
        
        # 4. 输出投影层
        # 将 ViG 的输出维度 (embed_dim) 映射到 DSDL 要求的维度 (2048)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.classifier_proj = nn.Linear(embed_dim, out_dim)

        # 初始化位置编码
        nn.init.normal_(self.pose_embedding, std=0.02)

    def forward(self, x):
        # x: [B, 1, 224, 224]
        
        # 1. Patchify
        # [B, 256, 16*16]
        x = self.patchifier(x) 
        # 将 [B, N, 16, 16] 变为 [B, N, 256]
        x = x.flatten(2)
        
        # 2. Embedding + Positional Encoding
        # [B, 256, embed_dim]
        x = self.patch_embedding(x)
        x = x + self.pose_embedding
        
        # 3. ViG Layers (提取拓扑特征)
        # [B, 256, embed_dim]
        x = self.blocks(x)
        
        # 4. Global Pooling (将 256 个节点聚合成 1 个图特征)
        # 类似于 ResNet 的 AdaptiveAvgPool
        # [B, embed_dim]
        x = F.adaptive_max_pool1d(x.transpose(1, 2), 1).squeeze(-1)
        # 或者使用平均池化: x = x.mean(dim=1)
        
        # 5. Projection to 2048
        x = self.bn(x)
        x = self.classifier_proj(x) # [B, 2048]
        
        return x
    
class DSDL(nn.Module):
    """Deep Semantic Dictionary Learning with dual-branch visual encoder.

    Visual encoder:
        Optical (4ch) -> ResNet101 -> 2048-d feature
        SAR (1ch)     -> ResNet101 -> 2048-d feature
        Fuse: f = f_opt + f_sar  (按导师要求：ResNet 输出后加和)

    The semantic dictionary part (W1/W2 + closed-form alpha) keeps the same.
    """

    def __init__(self, base_model_opt, num_classes, alpha, in_channel=300, num_scenes=3):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_scenes = num_scenes # 预设场景数 6分类问题暂设3个场景

        # Optical backbone
        self.features_opt = nn.Sequential(
            base_model_opt.conv1,
            base_model_opt.bn1,
            base_model_opt.relu,
            base_model_opt.maxpool,
            base_model_opt.layer1,
            base_model_opt.layer2,
            base_model_opt.layer3,
            base_model_opt.layer4,
        )

        self.features_sar = SARViGBackbone(
            in_channels=1, 
            patch_size=16, 
            embed_dim=320,  # 内部特征维度，可以调大如 512
            num_blocks=8,   # 堆叠层数，建议 6-12 层
            num_patches=256,# 对应 256x256 输入
            out_dim=2048,    # 必须是 2048，以便与 ResNet 光学分支对齐
            drop_path=0.1
        )

        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        # ===== semantic dictionary params (same as before) =====
        self.W1 = nn.Parameter(torch.zeros(size=(in_channel, 1024)))
        stdv1 = 1.0 / math.sqrt(self.W1.size(1))
        self.W1.data.uniform_(-stdv1, stdv1)

        self.relu = nn.LeakyReLU(0.2)

        self.W2 = nn.Parameter(torch.zeros(size=(1024, 2048)))
        stdv2 = 1.0 / math.sqrt(self.W2.size(1))
        self.W2.data.uniform_(-stdv2, stdv2)

        # Keep interface
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        # 场景感知模块
        # 1. 场景分类器
        self.scene_classifier = nn.Linear(2048, self.num_scenes)
        
        # 2. 注册全局场景共现频率矩阵 C_matrices [K, C, C]，不参与梯度更新
        self.register_buffer('C_matrices', torch.zeros(self.num_scenes, num_classes, num_classes))
        
        # 3. 动态字典融合的残差权重 (吸收共现信息的程度)
        self.beta = 0.1

    def forward(self, optical, sar, semantic_vectors, target=None):
        """Forward.

        optical: Tensor [B, 4, H, W]
        sar:     Tensor [B, 1, H, W]
        semantic_vectors: Tensor [C, D] or [B, C, D]
        """
        # semantic_vectors: [C, D]
        if semantic_vectors.dim() == 3:
            semantic_vectors = semantic_vectors[0]
        if semantic_vectors.dim() != 2:
            raise ValueError(
                f"语义向量应为2维 [num_classes, embedding_dim]，但得到了 {semantic_vectors.shape}"
            )

        if optical.size(1) != 4:
            raise ValueError(f"Optical 分支期望 4 通道输入，但得到了 {optical.size(1)}")
        if sar.size(1) != 1:
            raise ValueError(f"SAR 分支期望 1 通道输入，但得到了 {sar.size(1)}")

        # ===== 光学分支 =====
        f_opt = self.features_opt(optical)  # [B, 2048, H, W]
        # 使用 GMP 并展平空间维度
        f_opt = self.pooling_opt(f_opt).squeeze(-1).squeeze(-1) # [B, 2048]

        # ===== SAR分支 =====
        # 直接通过 ViG Backbone 得到 2048 维特征
        f_sar = self.features_sar(sar) # [B, 2048]

        # 直接加和
        feature = f_opt + f_sar

        # ===场景检测与共现矩阵更新===
        # 1. 预测场景分布
        scene_logits = self.scene_classifier(feature) # [B, K]
        scene_prob = F.softmax(scene_logits, dim=-1)  # [B, K]
        
        # 2. 训练期：利用真实的 Target 更新全局共现矩阵
        if self.training and target is not None:
            assigned_scenes = torch.argmax(scene_prob, dim=1)
            for i in range(feature.size(0)):
                s = assigned_scenes[i]
                y = target[i].unsqueeze(1)
                self.C_matrices[s] += torch.matmul(y, y.transpose(0, 1))

        # 3. 计算当前的场景共现概率矩阵 P^k
        # 取对角线元素 (即每个标签出现的总次数)，加 1e-8 防止除0
        c_diag = self.C_matrices.diagonal(dim1=1, dim2=2).unsqueeze(2) + 1e-8 # [K, C, 1]
        P_k = self.C_matrices / c_diag # [K, C, C] 条件概率
        
        # 4. 根据当前图片的场景分布，动态融合生成专属的 P^I
        # scene_prob: [B, K], P_k: [K, C, C] -> P_I: [B, C, C]
        P_I = torch.einsum('bk,kij->bij', scene_prob, P_k)

        # ===== semantic dictionary (same) =====
        semantic = torch.matmul(semantic_vectors, self.W1)  # [C,1024]
        semantic = self.relu(semantic)
        semantic = torch.matmul(semantic, self.W2)          # [C,2048]

        res_semantic = torch.matmul(semantic, self.W2.transpose(0, 1))
        res_semantic = self.relu(res_semantic)
        res_semantic = torch.matmul(res_semantic, self.W1.transpose(0, 1))

        # ===字典更新===
        B = feature.size(0)
        # 将基础字典扩展到 Batch 维度
        semantic_b = semantic.unsqueeze(0).expand(B, -1, -1) # [B, C, 2048]
        # 消息传递：P_I @ D
        message = torch.bmm(P_I, semantic_b) # [B, C, 2048]
        # 残差融合生成动态字典
        interacted_semantic = semantic_b + self.beta * message # [B, C, 2048]

        # ===动态闭式解计算===
        device = semantic.device
        # 计算 (D * D^T + alpha * I) 的批量求逆
        S_S_T = torch.bmm(interacted_semantic, interacted_semantic.transpose(1, 2)) # [B, C, C]
        eye_matrix = self.alpha * torch.eye(self.num_classes, device=device).unsqueeze(0).expand(B, -1, -1)
        inv_term = torch.linalg.inv(S_S_T + eye_matrix) # [B, C, C]
        # 计算 D * f
        S_f = torch.bmm(interacted_semantic, feature.unsqueeze(2)) # [B, C, 1]
        # 最终打分计算
        score = torch.bmm(inv_term, S_f).squeeze(2) # [B, C]

        return score, semantic_vectors, res_semantic, feature, semantic, scene_prob

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features_opt.parameters(), 'lr': lr * lrp},
            {'params': self.features_sar.parameters(), 'lr': lr},
            {'params': self.scene_classifier.parameters(), 'lr': lr},  # 场景分类器
            {'params': self.W1, 'lr': lr},                        
            {'params': self.W2, 'lr': lr},                        
        ]


def load_model(num_classes, alpha, pretrained=True, in_channel=300):
    """Build dual-branch model.

    - Optical branch: ResNet101 with conv1=4ch
    - SAR branch:     ResNet101 with conv1=1ch
    """
    if pretrained:
        weights = models.ResNet101_Weights.IMAGENET1K_V1
        base_opt = models.resnet101(weights=weights)
    else:
        base_opt = models.resnet101(weights=None)

    base_opt = modify_resnet_conv1(base_opt, in_channels=4)

    return DSDL(
        base_model_opt=base_opt,
        num_classes=num_classes,
        alpha=alpha,
        in_channel=in_channel,
    )


__all__ = ['DSDL', 'load_model', 'SARViGBackbone']
