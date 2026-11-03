import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder module for upsampling and feature fusion in a neural network.
    """

    def __init__(self, filters=None):
        super(Decoder, self).__init__()

        self.UpBlock3 = nn.Sequential(nn.ConvTranspose2d(filters[3], filters[2], padding=1, stride=2, kernel_size=(4, 4)),
                                      nn.BatchNorm2d(filters[2]),
                                      nn.LeakyReLU())

        self.UpBlock2 = nn.Sequential(nn.ConvTranspose2d(filters[2], filters[1], padding=1, stride=2, kernel_size=(4, 4)),
                                      nn.BatchNorm2d(filters[1]),
                                      nn.LeakyReLU())

        self.UpBlock1 = nn.Sequential(nn.ConvTranspose2d(filters[1], filters[0], padding=1, stride=2, kernel_size=(4, 4)),
                                      nn.BatchNorm2d(filters[0]),
                                      nn.LeakyReLU())

        self.final_conv = nn.Sequential(
            nn.Conv2d(filters[0]*4, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(),
            nn.Conv2d(filters[0], 1, kernel_size=1)
        )

        self.single_conv = nn.Conv2d(filters[0], 1, kernel_size=1)

    def forward(self, out_0, out_1, out_2, out_3):
        for upblock in [self.UpBlock3, self.UpBlock2, self.UpBlock1]:
            out_3 = upblock(out_3)
        for upblock in [self.UpBlock2, self.UpBlock1]:
            out_2 = upblock(out_2)
        for upblock in [self.UpBlock1]:
            out_1 = upblock(out_1)

        out_mask3 = self.single_conv(out_3)
        out_mask2 = self.single_conv(out_2)
        out_mask1 = self.single_conv(out_1)
        out_mask0 = self.single_conv(out_0)

        out_mask = self.final_conv(
            torch.cat([out_0, out_1, out_2, out_3], dim=1))

        return out_mask, [out_mask0, out_mask1, out_mask2, out_mask3]
