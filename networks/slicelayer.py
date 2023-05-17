import torch
import torch.nn as nn
import functools
import torchvision.models as models
import torch.nn.functional as F

def ud_pad(x, padding=1):
    _, _, H, W = x.shape
    idx = torch.arange(-W // 2, W // 2, device=x.device)
    up = x[:, :, :padding, idx].flip(2)
    down = x[:, :, -padding:, idx].flip(2)
    return torch.cat([up, x, down], dim=2)

def circle_pad(x, padding=(1, 1)):
    _,_,H,W = x.shape
    if padding[0] == 0 and padding[1] == 0:
        return x
    elif padding[0] == 0:
        return F.pad(x, pad=(padding[1], padding[1], 0, 0), mode='circular')
    else:
        idx = torch.arange(-W // 2, W // 2, device=x.device)
        up = x[:, :, :padding[0], idx].flip(2)
        down = x[:, :, -padding[0]:, idx].flip(2)
        return F.pad(torch.cat([up, x, down], dim=2), pad=(padding[1], padding[1], 0, 0), mode='circular')


class CIRCLE_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''

    def __init__(self, padding=(1,1)):
        super(CIRCLE_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return circle_pad(x, self.padding)

def wrap_circle_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0 & m.padding[0] == 0:
            continue
        w_pad = int(m.padding[1])
        h_pad = int(m.padding[0])
        m.padding = (0, 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(CIRCLE_PAD((h_pad,w_pad)), m)
        )

class CircPad(nn.Module):
    def __init__(self,padding):
        super().__init__()
        if isinstance(padding,int):
            self.h = padding
            self.w = padding
        else:
            self.h = padding[0]
            self.w = padding[1]
    def forward(self,x):
        _,_,H,W = x.shape
        assert H%2==0 and W%2==0
        if self.h == 0 and self.w == 0:
            return x
        elif self.h == 0:
            return F.pad(x,pad=(self.w,self.w,0,0),mode='circular')
        else:
            idx = torch.arange(-W//2,W//2,device=x.device)
            up = x[:,:,:self.h,idx].flip(2)
            down = x[:,:,-self.h:,idx].flip(2)
            return F.pad(torch.cat([up,x,down],dim=2),pad=(self.w,self.w,0,0),mode='circular')

def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)


class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''

    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)

class UD_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''

    def __init__(self, padding=1):
        super(UD_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return ud_pad(x, self.padding)

def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )

class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()

        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool

    def forward(self, x):

        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        features.append(x)  # 1/2
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        features.append(x)  # 1/4
        x = self.encoder.layer2(x)
        features.append(x)  # 1/8
        x = self.encoder.layer3(x)
        features.append(x)  # 1/16
        x = self.encoder.layer4(x)
        features.append(x)  # 1/32
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4
