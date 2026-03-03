import torch
import torch.nn as nn
from vig import ResBlock, ViGBlock
# from model.standard import opt_backbone


class NaNError(Exception):
    pass


class BN_Layer(nn.Module):
    def __init__(self, dim_z, tau=0.5, mu=True):
        super().__init__()
        self.dim_z = dim_z

        self.tau = torch.tensor(tau)  # tau : float in range (0,1)
        self.theta = torch.tensor(0.5, requires_grad=True)

        self.gamma1 = torch.sqrt(
            self.tau + (1 - self.tau) * torch.sigmoid(self.theta)
        )  # for mu
        self.gamma2 = torch.sqrt(
            (1 - self.tau) * torch.sigmoid((-1) * self.theta)
        )  # for var

        self.bn = nn.BatchNorm1d(dim_z)
        self.bn.bias.requires_grad = False
        self.bn.weight.requires_grad = True

        if mu:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma1)
        else:
            with torch.no_grad():
                self.bn.weight.fill_(self.gamma2)

    def forward(self, x):  # x:(batch_size,dim_z)
        x = self.bn(x)
        return x


class SimplePatchifier(nn.Module):
    def __init__(self, patch_size=8, num_patches=32):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 1, "Input must be a grayscale image with a single channel"

        unfolded = x.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        unfolded = unfolded.contiguous().view(
            B, C, -1, self.patch_size, self.patch_size
        )

        centers = unfolded[:, :, :, self.patch_size // 2, self.patch_size // 2]

        values, indices = torch.topk(centers.view(B, -1), self.num_patches, dim=1)

        unfolded = unfolded.view(B, -1, self.patch_size, self.patch_size)
        patches = unfolded.gather(
            1,
            indices.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, self.patch_size, self.patch_size),
        )
        if torch.isnan(patches).any():
            raise NaNError("have nan in model loss")

        patches = patches.unsqueeze(2)

        return patches


class VGNN(nn.Module):
    def __init__(
        self,
        in_features=1 * 16 * 16,
        out_feature=320,
        num_patches=196,
        num_ViGBlocks=2,
        num_edges=9,
        head_num=1,
        dimz=2048 * 2,
    ):
        super().__init__()

        self.patchifier = SimplePatchifier(
            patch_size=int(pow(in_features, 0.5)), num_patches=num_patches
        )
        # self.patch_embedding = TwoLayerNN(in_features)
        self.patch_embedding = nn.Sequential(
            nn.Linear(in_features, out_feature * 2),
            nn.BatchNorm1d(out_feature * 2),
            nn.LeakyReLU(),
            nn.Linear(out_feature * 2, out_feature),
            nn.BatchNorm1d(out_feature),
        )
        self.pose_embedding = nn.Parameter(torch.rand(num_patches, out_feature))

        self.blocks = nn.Sequential(
            *[ViGBlock(out_feature, num_edges, head_num) for _ in range(num_ViGBlocks)]
        )

        self.fc = nn.Linear(num_patches * out_feature, dimz, bias=False)

    def forward(self, x):
        x = self.patchifier(x)
        B, N, C, H, W = x.shape
        x = self.patch_embedding(x.view(B * N, -1)).view(B, N, -1)
        x = x + self.pose_embedding

        x = self.blocks(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


class SG_VGNN(nn.Module):
    def __init__(
        self,
        d,
        z,
        with_classifier=True,
        mtype="resnet",
        task="SOC",
        **kwargs
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d, track_running_stats=False),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d, track_running_stats=False),
            nn.ConvTranspose2d(
                d, d // 2, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(d // 2, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(
                d // 2, d // 4, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(d // 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(
                d // 4, d // 8, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(d // 8, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(
                d // 8, 1, kernel_size=4, stride=2, padding=1, bias=False
            ),
        )
        self.xi_bn = nn.BatchNorm2d(1, track_running_stats=False)
        self.f = 8
        self.d = d
        self.z = z
        self.fc21 = nn.Linear(self.z, d * self.f**2)
        self.with_classifier = with_classifier
        self.encoder = VGNN(
            in_features=1 * 16 * 16,
            out_feature=64,
            dimz=self.z * 2,
            num_patches=32,
        )
        if self.with_classifier:
            self.classifier = opt_backbone(mtype, task)
        self.bnlayer = BN_Layer(z)

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return h.flatten(start_dim=1), self.bnlayer(mu), logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            epsilon = torch.randn_like(mu)
            return mu + epsilon * torch.exp(logvar / 2)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        hi = self.reparameterize(mu, logvar)
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)
        xi = x.mul(x.mul(xi))
        xi = self.xi_bn(xi)

        if self.with_classifier:
            out = self.classifier(xi)
            return out, hi_projected, xi, mu, logvar
        else:
            return xi
