import torch
import torch.nn as nn

from .DURD_moudles.Decoder import Decoder
from .DURD_moudles.Encoder import Encoder
from .DURD_moudles.ASD import AdaptiveStructuredDecomposition


class Mix(nn.Module):
    def __init__(self, m=-0.8):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + \
            fea2 * (1-mix_factor.expand_as(fea2))
        return out


class DURD(nn.Module):
    'Deep Unrolled Residual Decomposition'

    def __init__(
            self,
            in_channel=1,
            base_channels=32,
            stage_num=6,
            tau=2.0,
            rank=None
    ):
        super(DURD, self).__init__()
        self.filters = [base_channels, base_channels *
                        2, base_channels * 4, base_channels * 8]

        self.encoder = Encoder(in_channel=in_channel, filters=self.filters)
        self.decoder = Decoder(filters=self.filters)

        self.ASD = AdaptiveStructuredDecomposition(
            stage_num=stage_num, in_channel=in_channel, tau=tau, rank=rank)
        self.mix = Mix()
        self.conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1)

    def forward(self, x):
        x = x if x.dim() != 3 else x.unsqueeze(0)

        L_out, S_out, X_recon_out = self.ASD(x)
        S_prior = self.conv1x1(S_out)

        res_fusion_feature = x - L_out
        out_0, out_1, out_2, out_3 = self.encoder(res_fusion_feature)
        F_dec, M = self.decoder(
            out_0, out_1, out_2, out_3)

        final_pred = self.mix(S_prior, F_dec)

        return L_out, S_out, X_recon_out, S_prior, M, F_dec, final_pred
