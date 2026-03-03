import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.BatchNorm1d(hidden_features),
            nn.LeakyReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.layer(x) + x


class ViGBlock(nn.Module):
    def __init__(self, in_features, num_edges=9, head_num=1):
        super().__init__()
        self.k = num_edges
        self.in_layer1 = TwoLayerNN(in_features)
        self.out_layer1 = TwoLayerNN(in_features)
        self.droppath1 = nn.Identity()  # DropPath(0)
        self.in_layer2 = TwoLayerNN(in_features, in_features * 4)
        self.out_layer2 = TwoLayerNN(in_features, in_features * 4)
        self.droppath2 = nn.Identity()  # DropPath(0)
        self.multi_head_fc = nn.Conv1d(
            in_features * 2, in_features, 1, 1, groups=head_num
        )

    def forward(self, x):
        B, N, C = x.shape

        sim = x @ x.transpose(-1, -2)
        graph = sim.topk(self.k, dim=-1).indices

        shortcut = x
        x = self.in_layer1(x.view(B * N, -1)).view(B, N, -1)

        # aggregation
        neibor_features = x[
            torch.arange(B).unsqueeze(-1).expand(-1, N).unsqueeze(-1), graph
        ]
        x = torch.stack([x, (neibor_features - x.unsqueeze(-2)).amax(dim=-2)], dim=-1)

        # update
        # Multi-head
        x = self.multi_head_fc(x.view(B * N, -1, 1)).view(B, N, -1)

        x = self.droppath1(
            self.out_layer1(F.leaky_relu(x).view(B * N, -1)).view(B, N, -1)
        )
        x = x + shortcut

        x = (
            self.droppath2(
                self.out_layer2(F.leaky_relu(self.in_layer2(x.view(B * N, -1)))).view(
                    B, N, -1
                )
            )
            + x
        )

        return x
