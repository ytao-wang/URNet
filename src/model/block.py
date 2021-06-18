import torch.nn.functional as F
import torch
import torch.nn as nn

from model.block_rfdn import E_RFDB


class CropLayer(nn.Module):

    # E.g., (-1, 0) means this layer should crop the first and last rows of the feature map.
    # And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class ACBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding_mode='zeros',
                 deploy=False):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        self.padding = kernel_size // 2
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=self.padding, dilation=dilation, groups=groups, bias=True,
                                        padding_mode=padding_mode)
        else:

            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=self.padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)

            center_offset_from_origin_border = self.padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)

            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:

            square_outputs = self.square_conv(input)

            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)

            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)

            return square_outputs + vertical_outputs + horizontal_outputs


class CFPB(nn.Module):
    def __init__(self, channel, res=False, fuse=True):
        super(CFPB, self).__init__()
        self.res = res
        self.fuse = fuse
        self.atrous_block3 = nn.Conv2d(channel, channel, 3, 1, padding=3, dilation=3)
        self.atrous_block6 = nn.Conv2d(channel, channel, 3, 1, padding=6, dilation=6)
        convs = []
        for i in range(2):
            convs.append(nn.Conv2d(2 * channel, channel, kernel_size=1, padding=0))
        self.convs = nn.ModuleList(convs)
        if self.fuse:
            self.fuse_conv = nn.Conv2d(channel * 3, channel, kernel_size=1, padding=0)

    def forward(self, x):
        ab3 = self.atrous_block3(x)
        ab3 = torch.cat([ab3, x], 1)
        ab3 = self.convs[0](ab3)

        ab6 = self.atrous_block6(ab3)
        ab6 = torch.cat([ab6, ab3], 1)
        ab6 = self.convs[1](ab6)

        out = torch.cat([x, ab3, ab6], 1)
        if self.fuse:
            out = self.fuse_conv(out)
        if self.res and self.fuse:
            out += x
        return out


class RFDBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ver=False, tail=False, add=False):
        super(RFDBBlock, self).__init__()
        if ver:
            block = [E_RFDB(in_channels, add=add)]
            if not tail:
                block.append(nn.Conv2d(in_channels, out_channels, 1, padding=0))
            self.block = nn.Sequential(*block)
        else:
            block = []
            if not tail:
                block.append(nn.Conv2d(in_channels, out_channels, 1, padding=0))
            block.append(E_RFDB(out_channels, add=add))
            self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class FDPRG(nn.Module):
    def __init__(self, channels, kernel_size=3, bias=True, scale=2):  # n_RG=4
        super(FDPRG, self).__init__()
        self.scale = scale
        self.w0 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w0.data.fill_(1.0)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)

        self.m1 = E_RFDB(channels)
        self.w_m1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w_m1.data.fill_(1.0)

        self.m2 = E_RFDB(channels)
        self.w_m2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w_m2.data.fill_(1.0)

        self.m3 = E_RFDB(channels)
        self.w_m3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w_m3.data.fill_(1.0)
        if self.scale != 3:
            self.aspp = CFPB(channels)
        else:
            self.cfpb = CFPB(channels)

        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=bias)

    def forward(self, x):
        res1 = self.m1(x)
        res1 = res1 + self.w_m1 * x

        res2 = self.m2(res1)
        res2 = res2 + self.w_m2 * res1
        res2 = res2 + self.w0 * x

        res3 = self.m3(res2)
        res3 = res3 + self.w_m3 * res2
        out = res3 + self.w1 * res1 + self.w2 * x
        if self.scale != 3:
            out = self.aspp(out)
        else:
            out = self.cfpb(out)
        out = self.conv(out)
        out += x
        return out


# Sampling method in ANRB
class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class ANRB(nn.Module):
    def __init__(self, in_channels, scale=1, psp_size=(1, 3, 6, 8)):
        super(ANRB, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_query = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)
        self.f_key = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)

        self.psp = PSPModule(psp_size)

        self.W = nn.Conv2d(in_channels=1, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        # query: Nx1xHxW -> Nx1xHW
        query = self.f_query(x).view(batch_size, 1, -1)
        # Nx1xHW -> NxHWx1
        query = query.permute(0, 2, 1)

        # key：Nx1xS
        key = self.f_key(x)
        key = self.psp(key)

        # value: Nx1xHW -> Nx1xS （S = 110）
        value = self.psp(self.f_value(x))
        # Nx1xS -> NxSx1 （S = 110）
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (1 ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, 1, h, w)
        context = self.W(context)
        context += x
        return context
