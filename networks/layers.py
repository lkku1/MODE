import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv3x3, self).__init__()

        self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, bias=bias)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels, bias)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ConvBnReLU, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels, bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

def upsample(x, scale=2):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)

# Based on https://github.com/sunset1995/py360convert
class Cube2Equirec(nn.Module):
    def __init__(self, face_w, equ_h, equ_w):
        super(Cube2Equirec, self).__init__()
        '''
        face_w: int, the length of each face of the cubemap
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        '''

        self.face_w = face_w
        self.equ_h = equ_h
        self.equ_w = equ_w


        # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
        self._equirect_facetype()
        self._equirect_faceuv()


    def _equirect_facetype(self):
        '''
        0F 1R 2B 3L 4U 5D
        '''
        tp = np.roll(np.arange(4).repeat(self.equ_w // 4)[None, :].repeat(self.equ_h, 0), 3 * self.equ_w // 8, 1)

        # Prepare ceil mask
        mask = np.zeros((self.equ_h, self.equ_w // 4), np.bool)
        idx = np.linspace(-np.pi, np.pi, self.equ_w // 4) / 4
        idx = self.equ_h // 2 - np.round(np.arctan(np.cos(idx)) * self.equ_h / np.pi).astype(int)
        for i, j in enumerate(idx):
            mask[:j, i] = 1
        mask = np.roll(np.concatenate([mask] * 4, 1), 3 * self.equ_w // 8, 1)

        tp[mask] = 4
        tp[np.flip(mask, 0)] = 5

        self.tp = tp
        self.mask = mask

    def _equirect_faceuv(self):

        lon = ((np.linspace(0, self.equ_w -1, num=self.equ_w, dtype=np.float32 ) +0.5 ) /self.equ_w - 0.5 ) * 2 *np.pi
        lat = -((np.linspace(0, self.equ_h -1, num=self.equ_h, dtype=np.float32 ) +0.5 ) /self.equ_h -0.5) * np.pi

        lon, lat = np.meshgrid(lon, lat)

        coor_u = np.zeros((self.equ_h, self.equ_w), dtype=np.float32)
        coor_v = np.zeros((self.equ_h, self.equ_w), dtype=np.float32)

        for i in range(4):
            mask = (self.tp == i)
            coor_u[mask] = 0.5 * np.tan(lon[mask] - np.pi * i / 2)
            coor_v[mask] = -0.5 * np.tan(lat[mask]) / np.cos(lon[mask] - np.pi * i / 2)

        mask = (self.tp == 4)
        c = 0.5 * np.tan(np.pi / 2 - lat[mask])
        coor_u[mask] = c * np.sin(lon[mask])
        coor_v[mask] = c * np.cos(lon[mask])

        mask = (self.tp == 5)
        c = 0.5 * np.tan(np.pi / 2 - np.abs(lat[mask]))
        coor_u[mask] = c * np.sin(lon[mask])
        coor_v[mask] = -c * np.cos(lon[mask])

        # Final renormalize
        coor_u = (np.clip(coor_u, -0.5, 0.5)) * 2
        coor_v = (np.clip(coor_v, -0.5, 0.5)) * 2

        # Convert to torch tensor
        self.tp = torch.from_numpy(self.tp.astype(np.float32) / 2.5 - 1)
        self.coor_u = torch.from_numpy(coor_u)
        self.coor_v = torch.from_numpy(coor_v)

        sample_grid = torch.stack([self.coor_u, self.coor_v, self.tp], dim=-1).view(1, 1, self.equ_h, self.equ_w, 3)
        self.sample_grid = nn.Parameter(sample_grid, requires_grad=False)

    def forward(self, cube_feat):

        bs, ch, h, w = cube_feat.shape
        assert h == self.face_w and w // 6 == self.face_w

        cube_feat = cube_feat.view(bs, ch, 1,  h, w)
        cube_feat = torch.cat(torch.split(cube_feat, self.face_w, dim=-1), dim=2)

        cube_feat = cube_feat.view([bs, ch, 6, self.face_w, self.face_w])
        sample_grid = torch.cat(bs * [self.sample_grid], dim=0)
        equi_feat = F.grid_sample(cube_feat, sample_grid, padding_mode="border", align_corners=True)

        return equi_feat.squeeze(2)


class Concat(nn.Module):
    def __init__(self, channels, **kwargs):
        super(Concat, self).__init__()
        self.conv = nn.Conv2d(channels*2, channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, equi_feat, c2e_feat):

        x = torch.cat([equi_feat, c2e_feat], 1)
        x = self.relu(self.conv(x))
        return x


# Based on https://github.com/Yeh-yu-hsuan/BiFuse/blob/master/models/FCRN.py
class BiProj(nn.Module):
    def __init__(self, channels, **kwargs):
        super(BiProj, self).__init__()

        self.conv_c2e = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))
        self.conv_e2c = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))
        self.conv_mask = nn.Sequential(nn.Conv2d(channels * 2, 1, kernel_size=1, padding=0),
                                       nn.Sigmoid())

    def forward(self, equi_feat, c2e_feat):
        aaa = self.conv_e2c(equi_feat)
        tmp_equi = self.conv_c2e(c2e_feat)
        mask_equi = self.conv_mask(torch.cat([aaa, tmp_equi], dim=1))
        tmp_equi = tmp_equi.clone() * mask_equi
        return equi_feat + tmp_equi


# from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
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

class StripPooling(nn.Module):

    def __init__(self, in_channels):
        super(StripPooling, self).__init__()
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))
        inter_channels = int(in_channels/16)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))

        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1,3), 1, (0,1), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3,1), 1, (1,0), bias=False),
                                     nn.BatchNorm2d(inter_channels))

        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, 1, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(1),
                                     nn.Sigmoid())

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)
        x2_1 = F.interpolate(self.conv2_1(self.pool3(x1)), (h, w), mode='bilinear', align_corners=True)
        x2_2 = F.interpolate(self.conv2_2(self.pool4(x1)), (h, w), mode='bilinear', align_corners=True)
        x2 = self.conv2_3(F.relu(x2_1 + x2_2))
        x3 = self.conv3(x2)

        return x * x3


class CEELayer(nn.Module):
    def __init__(self, in_channels,out_channels, SE=True):
        super(CEELayer, self).__init__()

        self.res_conv1 = nn.Conv2d(in_channels , out_channels, kernel_size=1, padding=0, bias=False)
        self.res_bn1 = nn.BatchNorm2d(out_channels)

        self.res_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.res_bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.SE = SE
        if self.SE:
            self.selayer = SELayer(in_channels)

        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, equi_feat, c2e_feat):

        x = torch.cat([equi_feat, c2e_feat], 1)
        x = self.relu(self.res_bn1(self.res_conv1(x)))
        shortcut = self.res_bn2(self.res_conv2(x))

        x = c2e_feat + shortcut
        x = torch.cat([equi_feat, x], 1)
        if self.SE:
            x = self.selayer(x)
        x = self.relu(self.conv(x))
        return x

# class refine(nn.Module):
#     def __init__(self, in_channels, channel, cube_h, equi_h, equi_w, scale):
#         super(refine, self).__init__()
#
#         # self.sqlayer = SELayer(self.in_channels)
#         # self.Strip = StripPooling(self.out_channels)
#
#         self.conv1 = nn.Sequential(nn.Conv2d(channel[0], channel[0]//scale, kernel_size=1, bias=False),
#                                    nn.BatchNorm2d(channel[0]//scale),
#                                    nn.PReLU(channel[0]//scale))
#         self.conv2 = nn.Sequential(nn.Conv2d(channel[1],  channel[1]//scale, kernel_size=1, bias=False),
#                                    nn.BatchNorm2d(channel[1]//scale),
#                                    nn.PReLU(channel[1]//scale))
#         self.conv3 = nn.Sequential(nn.Conv2d(channel[2],  channel[2]//scale, kernel_size=1, bias=False),
#                                    nn.BatchNorm2d(channel[2]//scale),
#                                    nn.PReLU(channel[2]//scale))
#         self.conv4 = nn.Sequential(nn.Conv2d(channel[3],  channel[3]//scale, kernel_size=1, bias=False),
#                                    nn.BatchNorm2d(channel[3]//scale),
#                                    nn.PReLU(channel[3]//scale))
#         self.conv5 = nn.Sequential(nn.Conv2d(channel[4],  channel[4]//scale, kernel_size=1, bias=False),
#                                    nn.BatchNorm2d(channel[4]//scale),
#                                    nn.PReLU(channel[4]//scale))
#
#         self.c2e = Cube2Equirec(cube_h, equi_h, equi_w)
#
#     def forward(self, equi_feat, cube_feat1, cube_feat2, cube_feat3, cube_feat4, cube_feat5):
#
#         face = equi_feat.shape[2]//2
#         cube_feat1 = F.interpolate(cube_feat1, size=(face, face), mode="bilinear", align_corners=False)
#         cube_feat1 = self.conv1(cube_feat1)
#         cube_feat1 = torch.cat(torch.split(cube_feat1, equi_feat.shape[0], dim=0), dim=-1)
#
#         cube_feat2 = F.interpolate(cube_feat2, size=(face, face), mode="bilinear", align_corners=False)
#         cube_feat2 = self.conv2(cube_feat2)
#         cube_feat2 = torch.cat(torch.split(cube_feat2, equi_feat.shape[0], dim=0), dim=-1)
#
#         cube_feat3 = F.interpolate(cube_feat3, size=(face, face), mode="bilinear", align_corners=False)
#         cube_feat3 = self.conv3(cube_feat3)
#         cube_feat3 = torch.cat(torch.split(cube_feat3, equi_feat.shape[0], dim=0), dim=-1)
#
#         cube_feat4 = F.interpolate(cube_feat4, size=(face, face), mode="bilinear", align_corners=False)
#         cube_feat4 = self.conv4(cube_feat4)
#         cube_feat4 = torch.cat(torch.split(cube_feat4, equi_feat.shape[0], dim=0), dim=-1)
#
#         cube_feat5 = F.interpolate(cube_feat5, size=(face, face), mode="bilinear", align_corners=False)
#         cube_feat5 = self.conv5(cube_feat5)
#         cube_feat5 = torch.cat(torch.split(cube_feat5, equi_feat.shape[0], dim=0), dim=-1)
#
#         feat = torch.cat([cube_feat1, cube_feat2, cube_feat3, cube_feat4, cube_feat5], dim=1)
#         feat = self.c2e(feat)
#         feat = torch.cat([equi_feat, feat],dim=1)
#         # x = self.sqlayer(equi_feat)
#         # x = self.res_conv1(x)
#         # x = self.Strip(x)
#
#         return feat


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Transformer_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Transformer_cascade(nn.Module):
    def __init__(self, emb_dims, num_patch, depth, num_heads):
        super(Transformer_cascade, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(emb_dims, eps=1e-6)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patch, emb_dims))
        nn.init.trunc_normal_(self.pos_emb, std=.02)
        for _ in range(depth):
            layer = Transformer_Block(emb_dims, num_heads=num_heads)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        hidden_states = x + self.pos_emb
        for i, layer_block in enumerate(self.layer):
            hidden_states = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        return encoded