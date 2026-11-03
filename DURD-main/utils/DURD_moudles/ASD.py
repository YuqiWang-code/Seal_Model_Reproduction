import torch
import torch.nn as nn


def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return nn.GroupNorm(g, num_channels)


class SubspaceLowrankModule(nn.Module):
    def __init__(self, in_channel=1, hidden_dim=32, rank=8, layers=2, norm="gn"):
        super().__init__()
        Norm = (lambda c: _group_norm(c)) if norm == "gn" else (
            lambda c: nn.BatchNorm2d(c))

        self.lift = nn.Sequential(
            nn.Conv2d(in_channel, hidden_dim,
                      kernel_size=3, padding=1, stride=1),
            Norm(hidden_dim),
            nn.ReLU(True),
        )
        self.down = nn.Conv2d(hidden_dim, rank, kernel_size=1,
                              stride=1, padding=0, bias=True)
        self.up = nn.Conv2d(rank, hidden_dim, kernel_size=1,
                            stride=1, padding=0, bias=True)

        refine = []
        for _ in range(layers):
            refine += [
                nn.Conv2d(rank, rank, kernel_size=3, padding=1, stride=1),
                nn.ReLU(True),
            ]
        self.refine = nn.Sequential(*refine) if refine else nn.Identity()

        self.proj_out = nn.Conv2d(
            hidden_dim, in_channel, kernel_size=3, padding=1, stride=1)

    def forward(self, D, T):
        x = D - T
        feat = self.lift(x)
        z = self.down(feat)
        z = self.refine(z)
        feat_hat = self.up(z)
        x_hat = self.proj_out(feat_hat)
        return x + x_hat


class StructuredSparseModule(nn.Module):
    def __init__(
        self,
        in_channel=1,
        hidden_dim=32,
        layers=6,
        stage_index=0,
        eps_max=0.2,
        m_min=0.05,
        tau=2.0
    ):
        super().__init__()
        self.stage_index = stage_index
        self.eps_max = float(eps_max)
        self.m_min = float(m_min)
        self.tau = float(tau)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, hidden_dim,
                      kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
        )

        blocks = []
        dilations = [1, 2, 3]
        for i in range(layers):
            d = dilations[i % len(dilations)]
            blocks += [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                          padding=d, dilation=d, stride=1),
                nn.ReLU(True),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(hidden_dim, in_channel,
                              kernel_size=3, padding=1, stride=1)

        mid = max(1, hidden_dim // 2)
        self.mask_net = nn.Sequential(
            nn.Conv2d(in_channel, mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(mid, in_channel, kernel_size=1, padding=0, stride=1),
        )

        # use raw epsilon then squash to (0, eps_max)
        # init to a small value: sigmoid(-3)~0.047 => eps ~ 0.0094 if eps_max=0.2
        self.raw_epsilon = nn.Parameter(
            torch.tensor(-3.0, dtype=torch.float32), requires_grad=True)

    def eps(self):
        return self.eps_max * torch.sigmoid(self.raw_epsilon)

    def forward(self, D, B, T):
        x = T + D - B

        fx = self.head(self.blocks(self.stem(x)))
        s = x - self.eps() * fx
        logits = self.mask_net(x) / self.tau
        M = self.m_min + (1.0 - self.m_min) * torch.sigmoid(logits)

        return M * s


class ConsistencyMergeModule(nn.Module):
    def __init__(self, in_channel=1, hidden_dim=32, layers=3, norm="gn", residual_scale=0.1):
        super().__init__()
        Norm = (lambda c: _group_norm(c)) if norm == "gn" else (
            lambda c: nn.BatchNorm2d(c))
        self.residual_scale = float(residual_scale)

        convs = [
            nn.Conv2d(in_channel, hidden_dim,
                      kernel_size=3, padding=1, stride=1),
            Norm(hidden_dim),
            nn.ReLU(True),
        ]
        for _ in range(layers):
            convs += [
                nn.Conv2d(hidden_dim, hidden_dim,
                          kernel_size=3, padding=1, stride=1),
                Norm(hidden_dim),
                nn.ReLU(True),
            ]
        convs.append(nn.Conv2d(hidden_dim, in_channel,
                     kernel_size=3, padding=1, stride=1))
        self.residual = nn.Sequential(*convs)

    def forward(self, B, T):
        x = B + T
        r = self.residual(x)
        return x + self.residual_scale * r


class AdaptiveDecompositionModule(nn.Module):
    def __init__(self, in_channel=1, hidden_dim=32, rank=8, stage_index=0, tau=2.0):
        super().__init__()
        self.lowrank = SubspaceLowrankModule(
            in_channel=in_channel, hidden_dim=hidden_dim, rank=rank, layers=2, norm="gn")
        self.sparse = StructuredSparseModule(in_channel=in_channel, hidden_dim=hidden_dim, layers=6, stage_index=stage_index,
                                             eps_max=0.2, m_min=0.05, tau=tau)
        self.merge = ConsistencyMergeModule(
            in_channel=in_channel, hidden_dim=hidden_dim, layers=3, norm="gn", residual_scale=0.1)

    def forward(self, D, T):
        B = self.lowrank(D, T)
        T = self.sparse(D, B, T)
        D = self.merge(B, T)
        return B, T, D


class AdaptiveStructuredDecomposition(nn.Module):
    def __init__(self, stage_num: int = 6, in_channel: int = 1, hidden_dim: int = 32, tau: float = 2.0, rank=None):
        super().__init__()
        self.stage_num = stage_num
        self.in_channel = in_channel
        self.hidden_dim = hidden_dim

        rank = max(4, hidden_dim // 4) if rank is None else rank

        self.decos = nn.ModuleList([
            AdaptiveDecompositionModule(
                in_channel=in_channel,
                hidden_dim=hidden_dim,
                rank=rank,
                stage_index=k,
                tau=tau
            )
            for k in range(stage_num)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, D):
        T = torch.zeros_like(D)
        B = torch.zeros_like(D)
        for k in range(self.stage_num):
            B, T, D = self.decos[k](D, T)
        return B, T, D
