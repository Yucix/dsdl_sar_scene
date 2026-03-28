import math
import torch
import torch.nn as nn
import torchvision.models as models
from vig import ViGBlock


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

class SARViGBranch(nn.Module):
    """
    SAR 分支：SLICO 动态超像素提取 + 节点特征编码 + ViG 图神经网络
    输入: SAR 预计算好的节点
    输出: 图级特征表示 [B, 2048] (与 Optical 分支对齐)
    """
    def __init__(self, patch_size=8, embed_dim=128, num_vig_blocks=2, num_segments=64, num_edges=9, head_num=1, drop_path=0.05):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_segments = num_segments
        self.num_edges = num_edges
        self.head_num = head_num

        # 节点嵌入: patch_size * patch_size -> embed_dim
        self.node_embed = nn.Sequential(
            nn.Linear(patch_size * patch_size, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # ViG Blocks 图传播
        self.vig_blocks = nn.ModuleList([
            ViGBlock(embed_dim, num_edges=num_edges, head_num=head_num, drop_path=drop_path)
            for _ in range(num_vig_blocks)
        ])

        # 读出层 (Readout): 将图特征对齐到 2048 维，以便与 ResNet101 融合
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, 512),  # 512 是一个经验值，可以调整
            nn.LeakyReLU(),
            nn.Linear(512, 2048)
        )

    def forward(self, nodes_batch, node_mask):
        # nodes_batch: [B, N, F]
        # node_mask:   [B, N]
        B, N, Fdim = nodes_batch.shape

        x = self.node_embed(nodes_batch.reshape(B * N, Fdim)).reshape(B, N, self.embed_dim)
        x = x * node_mask.unsqueeze(-1).float()

        for blk in self.vig_blocks:
            x = blk(x, node_mask=node_mask)

        mask_float = node_mask.unsqueeze(-1).float()
        sum_feat = (x * mask_float).sum(dim=1)
        num_feat = mask_float.sum(dim=1).clamp_min(1.0)
        graph_feat = sum_feat / num_feat

        out = self.out_proj(graph_feat)
        return out

class DSDL(nn.Module):
    """Deep Semantic Dictionary Learning with dual-branch visual encoder.

    Visual encoder:
        Optical (4ch) -> ResNet101 -> 2048-d feature
        SAR (1ch)     -> ResNet101 -> 2048-d feature
        Fuse: f = f_opt + f_sar  (按导师要求：ResNet 输出后加和)

    The semantic dictionary part (W1/W2 + closed-form alpha) keeps the same.
    """

    def __init__(
        self,
        base_model_opt,
        num_classes,
        alpha,
        in_channel=300,
        sar_patch_size=8,
        sar_embed_dim=128,
        sar_num_vig_blocks=2,
        sar_num_segments=64,
        sar_num_edges=9,
        sar_head_num=1,
        sar_drop_path=0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes

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

        # SAR backbone
        self.features_sar = SARViGBranch(
            patch_size=sar_patch_size,
            embed_dim=sar_embed_dim,
            num_vig_blocks=sar_num_vig_blocks,
            num_segments=sar_num_segments,
            num_edges=sar_num_edges,
            head_num=sar_head_num,
            drop_path=sar_drop_path,
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

    def forward(self, optical, sar, semantic_vectors, sar_nodes, node_mask):
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

        # ===== dual visual encoder =====
        f_opt = self.features_opt(optical)
        f_opt = self.pooling(f_opt).view(f_opt.size(0), -1)  # [B, 2048]

        # SAR 分支
        f_sar = self.features_sar(sar_nodes, node_mask)  # [B, 2048]

        # 光学 + SAR 融合
        feature = f_opt + f_sar

        # ===== semantic dictionary (same) =====
        semantic = torch.matmul(semantic_vectors, self.W1)  # [C,1024]
        semantic = self.relu(semantic)
        semantic = torch.matmul(semantic, self.W2)          # [C,2048]

        res_semantic = torch.matmul(semantic, self.W2.transpose(0, 1))
        res_semantic = self.relu(res_semantic)
        res_semantic = torch.matmul(res_semantic, self.W1.transpose(0, 1))

        device = semantic.device
        eye_matrix = self.alpha * torch.eye(self.num_classes, device=device)

        score = torch.matmul(
            torch.inverse(torch.matmul(semantic, semantic.transpose(0, 1)) + eye_matrix),
            torch.matmul(semantic, feature.transpose(0, 1))
        ).transpose(0, 1)

        return score, semantic_vectors, res_semantic, feature, semantic

    def get_config_optim(self, lr, lrp):
        # Important: optimize both backbones
        return [
            # 光学分支
            {'params': self.features_opt.parameters(), 'lr': lr * lrp},
            # SAR分支
            {'params': self.features_sar.parameters(), 'lr': lr},
            # semantic dictionary
            {'params': self.W1, 'lr': lr},
            {'params': self.W2, 'lr': lr},
        ]


def load_model(
    num_classes,
    alpha,
    pretrained=True,
    in_channel=300,
    sar_patch_size=8,
    sar_embed_dim=128,
    sar_num_vig_blocks=2,
    sar_num_segments=64,
    sar_num_edges=5,
    sar_head_num=1,
    sar_drop_path=0.0
):
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
        sar_patch_size=sar_patch_size,
        sar_embed_dim=sar_embed_dim,
        sar_num_vig_blocks=sar_num_vig_blocks,
        sar_num_segments=sar_num_segments,
        sar_num_edges=sar_num_edges,
        sar_head_num=sar_head_num,
        sar_drop_path=sar_drop_path,
    )


__all__ = ['DSDL', 'load_model']
