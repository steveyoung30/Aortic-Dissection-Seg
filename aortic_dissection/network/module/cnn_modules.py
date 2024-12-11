import torch
import torch.nn as nn

class ConvBnRelu3(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding, do_act=True, bias=True):
        super(ConvBnRelu3, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=ksize, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.do_act = do_act
        if do_act:
            self.act = nn.ReLU(inplace=True)
    def forward(self, input):
        out = self.bn(self.conv(input))
        if self.do_act:
            out = self.act(out)
        return out


class BottConvBnRelu3(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding, do_act=True, bias=True):
        super(BottConvBnRelu3, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=ksize, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.do_act = do_act
        if do_act:
            self.act = nn.ReLU(inplace=True)
    def forward(self, input):
        out = self.bn(self.conv(input))
        if self.do_act:
            out = self.act(out)
        return out

class ResidualBlock3(nn.Module):
    """ residual block with variable number of convolutions """

    def __init__(self, channels, ksize, padding, num_convs):
        super(ResidualBlock3, self).__init__()

        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(ConvBnRelu3(channels, channels, ksize, padding, do_act=True))
            else:
                layers.append(ConvBnRelu3(channels, channels, ksize, padding, do_act=False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):

        output = self.ops(input)
        return self.act(input + output)


class BottResidualBlock3(nn.Module):
    """ residual block with variable number of convolutions """

    def __init__(self, channels, ksize, padding, num_convs):
        super(BottResidualBlock3, self).__init__()

        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(BottConvBnRelu3(channels, channels, ksize, padding, do_act=True))
            else:
                layers.append(BottConvBnRelu3(channels, channels, ksize, padding, do_act=False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):

        output = self.ops(input)
        return self.act(input + output)

class InputBlock(nn.Module):
    """ input block of vb-net """

    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.bn(self.conv(input)))
        return out
        
class DownBlock(nn.Module):
    """ downsample block of v-net """

    def __init__(self, in_channels, num_convs, use_bottle_neck=False, kernel_size=[2, 2, 2], stride=[2, 2, 2]):
        super(DownBlock, self).__init__()
        out_channels = in_channels * 2
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=1)
        self.down_bn = nn.BatchNorm3d(out_channels)
        self.down_act = nn.ReLU(inplace=True)
        if use_bottle_neck:
            self.rblock = BottResidualBlock3(out_channels, 4, num_convs)
        else:
            self.rblock = ResidualBlock3(out_channels, 3, 1, num_convs)
    
    def forward(self, input):
        out = self.down_act(self.down_bn(self.down_conv(input)))
        out = self.rblock(out)
        return out     

class UpBlock(nn.Module):
    """ Upsample block of v-net """

    def __init__(self, in_channels, out_channels, num_convs, use_bottle_neck=False, kernel_size=[2, 2, 2], stride=[2, 2, 2]):
        super(UpBlock, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=kernel_size, stride=kernel_size, groups=1)
        self.up_bn = nn.BatchNorm3d(out_channels // 2)
        self.up_act = nn.ReLU(inplace=True)
        if use_bottle_neck:
            self.rblock = BottResidualBlock3(out_channels, 4, num_convs)
        else:
            self.rblock = ResidualBlock3(out_channels, 3, 1, num_convs)

    def forward(self, input, skip):
        out = self.up_act(self.up_bn(self.up_conv(input)))
        out = torch.cat((out, skip), 1)
        out = self.rblock(out)
        return out