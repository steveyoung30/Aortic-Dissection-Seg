import torch.nn as nn
import numpy as np
from aortic_dissection.network.module import InputBlock, UpBlock, DownBlock
# aortic_dissection.network.module import kaiming_weight_init
# aortic_dissection.network.module import gaussian_weight_init

class OutputBlock(nn.Module):
    """ output block of v-net

        The output is a list of foreground-background probability vectors.
        The length of the list equals to the number of voxels in the volume
    """

    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        out = self.act1(self.bn1(self.conv1(input)))
        out = self.conv2(out)
        out = self.softmax(out)
        return out

# def vnet_kaiming_init(net):

#     net.apply(kaiming_weight_init)

# def vnet_focal_init(net, obj_p):

#     net.apply(gaussian_weight_init)
#     # initialize bias such as the initial predicted prob for objects are at obj_p.
#     net.out_block.conv2.bias.data[1] = -np.log((1 - obj_p) / obj_p)

class Encoder(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1)
        self.down_64 = DownBlock(32, 2)
        self.down_128 = DownBlock(64, 3)
        self.down_256 = DownBlock(128, 3) 

    def forward(self, x):
        out = self.in_block(x)
        out = self.down_32(out)
        out = self.down_64(out)
        out = self.down_128(out)
        out = self.down_256(out)
        return out

class SegmentationNet(nn.Module):
    """ volumetric segmentation network """

    def __init__(self, in_channels, out_channels):
        super(SegmentationNet, self).__init__()
        self.in_block = InputBlock(in_channels, 16)
        self.down_32 = DownBlock(16, 1)
        self.down_64 = DownBlock(32, 2)
        self.down_128 = DownBlock(64, 3)
        self.down_256 = DownBlock(128, 3)
        self.up_256 = UpBlock(256, 256, 3)
        self.up_128 = UpBlock(256, 128, 3)
        self.up_64 = UpBlock(128, 64, 2)
        self.up_32 = UpBlock(64, 32, 1)
        self.out_block = OutputBlock(32, out_channels)

    def forward(self, input):
        out16 = self.in_block(input)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)
        out = self.up_256(out256, out128)
        out = self.up_128(out, out64)
        out = self.up_64(out, out32)
        out = self.up_32(out, out16)
        out = self.out_block(out)
        return out

    def max_stride(self):
        return [16, 16, 16]

if __name__ == "__main__":
    import torch
    from thop import profile, clever_format

    x=torch.randn(1, 1, 1, 128, 128, 256).cuda()
    net = SegmentationNet(1, 2).cuda()
    # out = net(x)
    # print(out.shape)

    flops, params = profile(net, inputs=x)
    flops, params = clever_format([flops, params])
    print(flops, params)
