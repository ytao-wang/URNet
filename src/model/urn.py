import torch
import torch.nn as nn

from model import common
from model.block import FDPRG, RFDBBlock, ANRB, ACBlock
from model.block_rfdn import pixelshuffle_block, E_RFDB


def make_model(args, parent=False):
    return URN(args)


class URN(nn.Module):
    def __init__(self, args):
        super(URN, self).__init__()
        nf = args.n_feats
        scale = args.scale[0]
        self.test_only = args.test_only
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = ACBlock(args.n_colors, nf, kernel_size=3)
        channels = [1, 2, 4, 8]
        down = []
        for p in range(4):
            if p == 3:
                down.append(RFDBBlock(nf // channels[p], nf // channels[p], ver=True, tail=True))
            else:
                down.append(RFDBBlock(nf // channels[p], nf // channels[p + 1], ver=True))
        self.down = nn.ModuleList(down)

        up = []
        for p in range(4):
            if p == 3:
                up.append(E_RFDB(nf // channels[3 - p], nf))
            else:
                up.append(FDPRG(nf // channels[3 - p], nf // channels[3 - p], scale=scale))
        self.up = nn.ModuleList(up)

        self.conv = common.default_conv(nf, nf, kernel_size=3)
        self.anrb = ANRB(nf)

        self.tail_up = pixelshuffle_block(nf, args.n_colors, upscale_factor=scale)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        input_x = x

        down_out = []
        for i in range(4):
            x = self.down[i](x)
            down_out.append(x)

        x = self.up[0](x)
        for i in range(1, 4):
            x = torch.cat([down_out[-1 - i], x], dim=1)
            x = self.up[i](x)

        x = self.conv(x)
        x += input_x
        x = self.anrb(x)

        sr_out = self.tail_up(x)
        sr_out = self.add_mean(sr_out)

        return sr_out

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

