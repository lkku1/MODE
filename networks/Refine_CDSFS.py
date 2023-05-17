from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from .mobilenet import *
from .layers import Conv3x3, ConvBnReLU, upsample
from .slicelayer import Resnet, wrap_lr_pad
from collections import OrderedDict
import torch.nn.functional as F
from.consist import CDSF, Fusion
from .P2E import pers2equi

class CDSFS(nn.Module):
    """ UniFuse Model: Resnet based Euqi Encoder and Cube Encoder + Euqi Decoder """

    def __init__(self, num_layers, equi_h, equi_w):
        super(CDSFS, self).__init__()

        self.num_layers = num_layers
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.cube_h = equi_h//2

        self.patch_size = 256
        self.npatches = 13
        self.nrows = 3
        self.fov = 105

        self.net = {
                    18: 'resnet18',
                    34: 'resnet34',
                    50: 'resnet50'
                    }

        if (self.num_layers < 18):
            self.equi_encoder = mobilenet_v2(pretrained=True)
        else:
            self.equi_encoder = Resnet(self.net[self.num_layers], pretrained=True)

        equi_c1, equi_c2, equi_c3, equi_c4, equi_c5 = 64, 64, 128, 256, 512
        self.num_ch_dec = np.array([equi_c1, equi_c2, equi_c3, equi_c4, equi_c5])

        if self.num_layers < 18:
            equi_c1, equi_c2, equi_c3, equi_c4, equi_c5 = 16, 24, 32, 96, 320
            self.num_ch_dec = np.array([equi_c1, equi_c2, equi_c3, equi_c4, equi_c5])

        self.fusion_net = CDSF(equi_c4, equi_c2, 96)
        self.fusion = Fusion(self.num_ch_dec[4], 96, self.equi_h//32)

        self.equi_dec_convs = OrderedDict()
        self.equi_dec_convs["upconv_5"] = ConvBnReLU(equi_c5, self.num_ch_dec[4])

        self.equi_dec_convs["deconv_4"] = ConvBnReLU(self.num_ch_dec[4] + equi_c4, self.num_ch_dec[4])
        self.equi_dec_convs["upconv_4"] = ConvBnReLU(self.num_ch_dec[4], self.num_ch_dec[3])

        self.equi_dec_convs["deconv_3"] = ConvBnReLU(self.num_ch_dec[3] + equi_c3, self.num_ch_dec[3])
        self.equi_dec_convs["upconv_3"] = ConvBnReLU(self.num_ch_dec[3], self.num_ch_dec[2])

        self.equi_dec_convs["deconv_2"] = ConvBnReLU(self.num_ch_dec[2] + equi_c2, self.num_ch_dec[2])
        self.equi_dec_convs["upconv_2"] = ConvBnReLU(self.num_ch_dec[2], self.num_ch_dec[1])

        self.equi_dec_convs["deconv_1"] = ConvBnReLU(self.num_ch_dec[1] + equi_c1, self.num_ch_dec[1])
        self.equi_dec_convs["upconv_1"] = ConvBnReLU(self.num_ch_dec[1], self.num_ch_dec[0])

        self.equi_dec_convs["deconv_0"] = ConvBnReLU(self.num_ch_dec[0], self.num_ch_dec[0])
        self.equi_dec_convs["depthconv_0"] = Conv3x3(self.num_ch_dec[0], 1)

        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))


    def forward(self, input_equi_image, depth_patch_map, roll_idx, flip):

        equi_enc_feat1, equi_enc_feat2, equi_enc_feat3, equi_enc_feat4, equi_enc_feat5 = self.equi_encoder(input_equi_image)

        depth_patch_map = torch.unsqueeze(depth_patch_map, dim=1)

        depth_patch = torch.cat(torch.split(torch.unsqueeze(depth_patch_map, dim=-1), self.patch_size, dim=-2), dim=-1) #bs, 1, ps, ps, 18

        depth_init = pers2equi(depth_patch, self.fov, self.nrows, (self.patch_size, self.patch_size), (self.equi_h//4, self.equi_w//4), "pred_512_13", roll_idx, flip) #bs, 1, 512, 1024, 18

        css_featue, csss = self.fusion_net(equi_enc_feat2, equi_enc_feat4, depth_init)

        equi_enc_feat5 = self.fusion(equi_enc_feat5, css_featue)
        outputs = {}
        equi_x = upsample(self.equi_dec_convs["upconv_5"](equi_enc_feat5))
        equi_x = self.equi_dec_convs["deconv_4"](torch.cat([equi_enc_feat4, equi_x], dim=1))

        equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_x))
        equi_x = self.equi_dec_convs["deconv_3"](torch.cat([equi_enc_feat3, equi_x], dim=1))

        equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))
        equi_x = self.equi_dec_convs["deconv_2"](torch.cat([equi_enc_feat2, equi_x], dim=1))

        equi_x = upsample(self.equi_dec_convs["upconv_2"](equi_x))
        equi_x = self.equi_dec_convs["deconv_1"](torch.cat([equi_enc_feat1, equi_x], dim=1))

        equi_x = upsample(self.equi_dec_convs["upconv_1"](equi_x))
        equi_x = self.equi_dec_convs["deconv_0"](equi_x)
        equi_depth = self.equi_dec_convs["depthconv_0"](equi_x)

        outputs["pred_depth"] = F.relu(equi_depth)
        outputs['coarse_scene_structure'] = csss


        return outputs
