import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class TwoLayerNN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            # 【核心修改】将 BatchNorm1d 替换为 LayerNorm
            nn.LayerNorm(hidden_features), 
            nn.LeakyReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.layer(x) + x


class ViGBlock(nn.Module):
    def __init__(self, in_features, num_edges=9, head_num=1, drop_path=0.05):
        super().__init__()
        self.k = num_edges
        self.in_layer1 = TwoLayerNN(in_features)
        self.out_layer1 = TwoLayerNN(in_features)
        self.droppath1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.in_layer2 = TwoLayerNN(in_features, in_features * 4)
        self.out_layer2 = TwoLayerNN(in_features, in_features * 4)
        self.droppath2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.multi_head_fc = nn.Conv1d(
            in_features * 2, in_features, 1, 1, groups=head_num
        )

    def forward(self, x, node_mask=None):
        """
        x: [B, N, C]
        node_mask: [B, N], True/1 表示真实节点，False/0 表示 padding 节点
        """
        B, N, C = x.shape
        device = x.device

        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            node_mask = node_mask.bool()

        shortcut = x

         # MLP 前向
        x = self.in_layer1(x.reshape(B * N, -1)).reshape(B, N, -1)

        # padding 节点特征先清零，避免后续残差污染
        x = x * node_mask.unsqueeze(-1)

        # ===== masked graph construction =====
        sim = x @ x.transpose(-1, -2)
        
        # 只有“真实节点->真实节点”允许连边
        valid_pair = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)  # [B, N, N]

        # 屏蔽非法位置
        sim = sim.masked_fill(~valid_pair, float("-inf"))

        # 对于 padding query 节点，整行可能全是 -inf，topk 会有问题
        # 这里给 padding query 节点的对角线补 0，保证 topk 可运行
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # [1, N, N]
        pad_query = ~node_mask  # [B, N]
        sim = sim.masked_fill(pad_query.unsqueeze(-1) & eye, 0.0)

        # 防止 k > 有效节点数
        num_valid = node_mask.sum(dim=1)                  # [B]
        k_eff = min(self.k, max(1, int(num_valid.max().item())))

        graph = sim.topk(k_eff, dim=-1).indices          # [B, N, k_eff]
        
        # ===== aggregation =====
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k_eff)
        neighbor_features = x[batch_idx, graph]          # [B, N, k_eff, C]

        center = x.unsqueeze(-2)                         # [B, N, 1, C]
        agg = (neighbor_features - center).amax(dim=-2) # [B, N, C]

        x = torch.stack([x, agg], dim=-1)                # [B, N, C, 2]
        x = self.multi_head_fc(x.reshape(B * N, -1, 1)).reshape(B, N, -1)

        x = self.droppath1(
            self.out_layer1(F.leaky_relu(x).reshape(B * N, -1)).reshape(B, N, -1)
        )
        x = x + shortcut

        x = (
            self.droppath2(
                self.out_layer2(
                    F.leaky_relu(self.in_layer2(x.reshape(B * N, -1)))
                ).reshape(B, N, -1)
            )
            + x
        )

        # 最后再次清零 padding 节点，避免残差把假节点激活
        x = x * node_mask.unsqueeze(-1)
        return x