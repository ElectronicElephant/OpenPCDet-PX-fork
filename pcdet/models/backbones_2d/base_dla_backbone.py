# ------------------------------------------------------------------------------
# Smoke mono3D
# backbone detector head
# ------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import math

from ..model_utils.DCNv2.dcn_v2 import DCN
from ..model_utils.weight_process import _fill_up_weights, _HEAD_NORM_SPECS

class DeformConv(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 norm_func):
        super(DeformConv, self).__init__()

        self.norm = norm_func(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.deform_conv = DCN(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1,
                               dilation=1,
                               deformable_groups=1)
    def forward(self, x):
        x = self.deform_conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

DLA34DCN = {
    "levels": [1, 1, 1, 2, 2, 1],
    "channels": [16, 32, 64, 128, 256, 512],
    "block": "BasicBlock"
}

# -----------------------------------------------------------------------------
# DLA models
# -----------------------------------------------------------------------------

def get_base_model(model, norm_func):
    model = DLABase(levels=model["levels"],
                    channels=model["channels"],
                    block=eval(model["block"]),
                    norm_func=norm_func)
    return model


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_func,
                 stride=1,
                 dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias=False,
                               dilation=dilation)
        self.norm1 = norm_func(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=dilation,
                               bias=False,
                               dilation=dilation
                               )
        self.norm2 = norm_func(out_channels)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.relu(out)

        return out


class Tree(nn.Module):
    def __init__(self,
                 level,
                 block,
                 in_channels,
                 out_channels,
                 norm_func,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False
                 ):
        super(Tree, self).__init__()

        if root_dim == 0:
            root_dim = 2 * out_channels

        if level_root:
            root_dim += in_channels

        if level == 1:
            self.tree1 = block(in_channels,
                               out_channels,
                               norm_func,
                               stride,
                               dilation=dilation)

            self.tree2 = block(out_channels,
                               out_channels,
                               norm_func,
                               stride=1,
                               dilation=dilation)
        else:
            new_level = level - 1
            self.tree1 = Tree(new_level,
                              block,
                              in_channels,
                              out_channels,
                              norm_func,
                              stride,
                              root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation,
                              root_residual=root_residual)

            self.tree2 = Tree(new_level,
                              block,
                              out_channels,
                              out_channels,
                              norm_func,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation,
                              root_residual=root_residual)
        if level == 1:
            self.root = Root(root_dim,
                             out_channels,
                             norm_func,
                             root_kernel_size,
                             root_residual)

        self.level_root = level_root
        self.root_dim = root_dim
        self.level = level

        self.downsample = None
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)

        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          bias=False),

                norm_func(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        if children is None:
            children = []

        if self.downsample:
            bottom = self.downsample(x)
        else:
            bottom = x

        if self.project:
            residual = self.project(bottom)
        else:
            residual = bottom

        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)

        if self.level == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class Root(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_func,
                 kernel_size,
                 residual):
        super(Root, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=1,
                              bias=False,
                              padding=(kernel_size - 1) // 2)

        self.norm = norm_func(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.norm(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x

class DLABase(nn.Module):
    def __init__(self,
                 levels,
                 channels,
                 block=BasicBlock,
                 residual_root=False,
                 norm_func=nn.BatchNorm2d,
                 ):
        super(DLABase, self).__init__()

        self.channels = channels
        self.level_length = len(levels)

        self.base_layer = nn.Sequential(nn.Conv2d(3,
                                                  channels[0],
                                                  kernel_size=7,
                                                  stride=1,
                                                  padding=3,
                                                  bias=False),

                                        norm_func(channels[0]),

                                        nn.ReLU(inplace=True))

        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0],
                                       norm_func=norm_func)

        self.level1 = self._make_conv_level(channels[0], channels[1], levels[0],
                                       norm_func=norm_func, stride=2)

        self.level2 = Tree(level=levels[2],
                           block=block,
                           in_channels=channels[1],
                           out_channels=channels[2],
                           norm_func=norm_func,
                           stride=2,
                           level_root=False,
                           root_residual=residual_root)

        self.level3 = Tree(level=levels[3],
                           block=block,
                           in_channels=channels[2],
                           out_channels=channels[3],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

        self.level4 = Tree(level=levels[4],
                           block=block,
                           in_channels=channels[3],
                           out_channels=channels[4],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

        self.level5 = Tree(level=levels[5],
                           block=block,
                           in_channels=channels[4],
                           out_channels=channels[5],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1, norm_func=nn.BatchNorm2d):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                norm_func(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)

        for i in range(self.level_length):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)

        return y


class DLAUp(nn.Module):
    def __init__(self,
                 startp,
                 channels,
                 scales,
                 in_channels=None,
                 norm_func=nn.BatchNorm2d):
        super(DLAUp, self).__init__()

        self.startp = startp

        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)

        scales = np.array(scales, dtype=int)

        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self,
                    'ida_{}'.format(i),
                    IDAUp(in_channels[j:],
                          channels[j],
                          scales[j:] // scales[j],
                          norm_func))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class IDAUp(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channel,
                 up_f,  # todo: what is up_f here?
                 norm_func):
        super(IDAUp, self).__init__()

        for i in range(1, len(in_channels)):
            in_channel = in_channels[i]
            f = int(up_f[i])
            proj = nn.Sequential(
                    nn.Conv2d(in_channel,out_channel,kernel_size=1, stride=1, bias=False),
                    norm_func(out_channel),
                    nn.ReLU())

            node = nn.Sequential(nn.Conv2d(out_channel*2, out_channel,
                          kernel_size= 3, stride=1,
                          padding= 3 // 2, bias=False),
                norm_func(out_channel),
                nn.ReLU())
            up = nn.ConvTranspose2d(out_channel,
                                    out_channel,
                                    kernel_size=f * 2,
                                    stride=f,
                                    padding=f // 2,
                                    output_padding=0,
                                    groups=out_channel,
                                    bias=False)
            _fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(torch.cat([layers[i], layers[i-1]], 1))

class DLA(nn.Module):
    def __init__(self,
                 model_cfg, input_channels, last_level=5):
        super(DLA, self).__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('DOWN_RATIO', None) is not None:
            assert self.model_cfg.DOWN_RATIO in [2, 4, 8, 16]
        else:
            raise ValueError("DOWN_RATIO info not found in model_cfg")

        if self.model_cfg.get('NORM_FUNC', None) is None:
            raise ValueError("NORM_FUNC info not found in model_cfg")

        base_name = DLA34DCN
        norm_func = _HEAD_NORM_SPECS[self.model_cfg.NORM_FUNC]
        channels = base_name['channels']

        self.first_level = int(np.log2(self.model_cfg.DOWN_RATIO))
        self.last_level = last_level
        out_channel = channels[self.first_level]

        self.out_channels = out_channel
        self.num_bev_features = out_channel
        # self.base = globals()[base_name]
        self.base = get_base_model(base_name, norm_func)

        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(startp=self.first_level,
                            channels=channels[self.first_level:],
                            scales=scales,
                            norm_func=norm_func)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        up_scales = [2 ** i for i in range(self.last_level - self.first_level)]
        self.ida_up = IDAUp(in_channels=channels[self.first_level:self.last_level],
                            out_channel=out_channel,
                            up_f=up_scales,
                            norm_func=norm_func)

    def forward(self, x):
        x = self.base(x['images'])
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        # todo: this can be further cleaned
        return y[-1]