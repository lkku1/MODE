import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .slicelayer import LR_PAD, UD_PAD
from .layers import ConvBnReLU

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Strip_operator(nn.Module):
    def __init__(self, dim, type, global_kernel_size):
        super().__init__()
        self.type = type  # H or W
        self.dim = dim
        self.global_kernel_size = global_kernel_size
        self.kernel_size = (global_kernel_size, 1) if self.type == 'H' else (1, global_kernel_size)
        self.padding = global_kernel_size//2
        self.gcc_conv = nn.Conv2d(dim, dim, kernel_size=self.kernel_size, padding=(0, 0), groups=dim)
        if self.type=='H':
            # self.pe = nn.Parameter(torch.randn(1, dim, self.global_kernel_size, 1))
            self.pad = UD_PAD(self.padding)
        elif self.type=='W':
            # self.pe = nn.Parameter(torch.randn(1, dim, 1, self.global_kernel_size))
            self.pad = LR_PAD(self.padding)
        # trunc_normal_(self.pe, std=.02)

    def forward(self, x):
        # shape = x.shape
        # x = x + self.pe.expand(1, self.dim, shape[2], shape[3])
        x = self.pad(x)
        x = self.gcc_conv(x)

        return x

class Strip_Conv(nn.Module):
    def __init__(self, hdim, out):
        super().__init__()
        # self.conv = nn.Conv2d(out, out, kernel_size=(1, 1), bias=False)
        # self.gcc_H = Strip_operator(out, 'H', 128)
        # self.gcc_W = Strip_operator(out, 'W', 256)
        self.gcc_H = Strip_operator(out, 'H', hdim)
        self.gcc_W = Strip_operator(out, 'W', 2*hdim)
        self.norm = nn.LayerNorm(out, eps=1e-6)
        self.pwconv1 = nn.Linear(out, 2 * out)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * out, out)
        self.act2 = nn.Sigmoid()

        self.drop_out = nn.Dropout2d()

    def forward(self, x):
        # input = self.conv(x)
        input = x
        x_H, x_W = self.gcc_H(input), self.gcc_W(input)
        x = (x_H[:,:,1:,:]+x_W[:,:,:,1:])
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, H, W) -> (N, H, W, C)
        y = self.norm(x)
        y = self.pwconv1(y)
        y = self.act(y)
        y = self.pwconv2(y)
        y = self.act2(y) * x
        y = y.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_out(y)
        return x


class CDSF(nn.Module):
    def __init__(self, g_dim, l_dim, dim):
        super(CDSF, self).__init__()

        self.conv_dep = nn.Sequential(
                nn.Conv2d(13, dim, kernel_size=(3, 3), stride=(1,1), padding=(1, 1), bias=False),  #16   32
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
        )
        self.selayer = SELayer(dim)

        self.conv_ff = nn.Sequential(
            nn.Conv2d(l_dim + l_dim + dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # 96  128
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.upconv = nn.Sequential(
            nn.Conv2d(g_dim, l_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(l_dim),
            nn.ReLU(inplace=True),
        )

        self.pre_dep = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), bias=False),
            nn.ELU(inplace=True)
        )
    def forward(self, l_f, g_f, dep):

        # dep = torch.squeeze(dep, dim=-1)
        dep = torch.squeeze(dep.permute(0, 4, 2, 3, 1), dim=-1)
        g_f = self.upconv(F.interpolate(g_f, size=l_f.shape[2:]))
        gl_f = torch.cat([g_f, l_f], dim=1)   #bs, 64, h/4, w/4        gl-f

        dep_feat = self.selayer(self.conv_dep(dep))  #bs, 64, h/4, w/4
        dep_fuse = self.conv_ff(torch.cat([dep_feat, gl_f], dim=1))

        cdf = dep_feat + dep_fuse
        cds = self.pre_dep(cdf)

        return cdf, cds

class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, (3, 3), stride=(2, 2), padding=(1, 1), bias=False,
                  padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), stride=(1, 1), padding=(1, 1), bias=False,
                               padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.down = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1), stride=(2,2), bias=False),
                nn.BatchNorm2d(out_dim),
            )

    def forward(self, feature):

        out = self.conv1(feature)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.down(feature)
        out += identity
        out = self.relu(out)
        return out

# class Double_dec(nn.Module):
#     def __init__(self, dec_dim, enc_dim):
#         super(Double_dec, self).__init__()
#
#         self.convde = nn.Sequential(
#             nn.Conv2d(enc_dim + dec_dim, dec_dim, (3, 3), stride=(1, 1), padding=(1, 1), bias=False,
#                       padding_mode='zeros'),
#             nn.BatchNorm2d(dec_dim),
#             nn.ReLU(inplace=True))
#
#         self.seg = nn.Sequential(
#             nn.Conv2d(dec_dim, dec_dim // 2, (3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode='zeros'),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(dec_dim // 2, dec_dim // 2, (4, 4), stride=(2, 2), padding=(1, 1), bias=False,
#                                padding_mode='zeros'),
#             nn.ReLU(True),
#             nn.Conv2d(dec_dim // 2, dec_dim // 4, (3, 3), stride=(1, 1), padding=(1, 1), bias=False,
#                       padding_mode='zeros'),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(dec_dim // 4, dec_dim // 4, (4, 4), stride=(2, 2), padding=(1, 1), bias=False,
#                                padding_mode='zeros'),
#             nn.ReLU(True),
#             nn.Conv2d(dec_dim // 4, 1, (3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode='zeros'),
#             nn.Sigmoid()
#         )
#
#     def forward(self, rgb_end, gui_dep):
#         dec = self.convde(torch.cat([rgb_end, gui_dep], dim=1))
#         seg = self.seg(dec)
#
#         return dec, seg

class Fusion(nn.Module):
    def __init__(self, enc_dim, dim, hdim):
        super(Fusion, self).__init__()

        self.convde = nn.Sequential(
            nn.Conv2d(enc_dim + 128, enc_dim, (3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode='zeros'),
            nn.BatchNorm2d(enc_dim),
            nn.ReLU(inplace=True))

        self.depth = nn.Sequential(
            BasicBlock(dim, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128))

        self.convst = Strip_Conv(hdim, enc_dim)


    def forward(self, rgb_end, gui_dep):
        depth_feature = self.depth(gui_dep)
        dec = self.convde(torch.cat([rgb_end, depth_feature], dim=1))
        dec = self.convst(dec)

        return dec

class Fusion2(nn.Module):
    def __init__(self, enc_dim, dim, hdim):
        super(Fusion2, self).__init__()

        self.convde = nn.Sequential(
            nn.Conv2d(enc_dim + 128, enc_dim, (3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode='zeros'),
            nn.BatchNorm2d(enc_dim),
            nn.ReLU(inplace=True))

        self.depth = nn.Sequential(
            BasicBlock(dim, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128))


    def forward(self, rgb_end, gui_dep):
        depth_feature = self.depth(gui_dep)
        dec = self.convde(torch.cat([rgb_end, depth_feature], dim=1))

        return dec

class FastUpconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=False), nn.BatchNorm2d(out_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3), bias=False), nn.BatchNorm2d(out_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2), bias=False), nn.BatchNorm2d(out_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2), bias=False), nn.BatchNorm2d(out_channels))
        self.ps = nn.PixelShuffle(2)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x1 = self.conv1(nn.functional.pad(x, (1, 1, 1, 1)))
        x2 = self.conv2(nn.functional.pad(x, (1, 1, 0, 1)))
        x3 = self.conv3(nn.functional.pad(x, (0, 1, 1, 1)))
        x4 = self.conv4(nn.functional.pad(x, (0, 1, 0, 1)))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.ps(x)
        x = self.act(x)
        return x


class Double_dec(nn.Module):
    def __init__(self, dec_dim, enc_dim):
        super(Double_dec, self).__init__()

        self.convde = nn.Sequential(
            nn.Conv2d(enc_dim + dec_dim, dec_dim, (3, 3), stride=(1, 1), padding=(1, 1), bias=False,
                      padding_mode='zeros'),
            nn.BatchNorm2d(dec_dim),
            nn.ReLU(inplace=True))

        self.seg = nn.Sequential(
            FastUpconv(dec_dim, dec_dim//4),
            ConvBnReLU(dec_dim//4, dec_dim//4),
            FastUpconv(dec_dim//4, dec_dim // 8),
            ConvBnReLU(dec_dim // 8, dec_dim // 8),
            nn.Conv2d(dec_dim // 8, 1, (3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode='zeros'),
            nn.Sigmoid()
        )

    def forward(self, rgb_end, gui_dep):
        dec = self.convde(torch.cat([rgb_end, gui_dep], dim=1))
        seg = self.seg(dec)

        return dec, seg
