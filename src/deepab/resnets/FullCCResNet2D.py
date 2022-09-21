"""PyTorch implementation of a ResNet with 2D CNNs

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""
import torch.nn as nn
import torch.nn.functional as F

from src.deepab.resnets.CrissCrossResNet2D import RCCAModule


class FullCCResNet2D(nn.Module):
    def __init__(self, in_channels, channels, num_blocks, kernel_size=5):

        super(FullCCResNet2D, self).__init__()

        self.activation = F.relu
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.channels = channels
        self.num_blocks = num_blocks
        self.device = None

        self.conv1 = nn.Conv2d(self.in_channels,
                               self.channels,
                               kernel_size=(kernel_size, kernel_size),
                               stride=(1, 1),
                               padding=(kernel_size // 2, kernel_size // 2),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels)

        self.layers = []
        for i in range(self.num_blocks):
            cc_block = RCCAModule(in_channels=self.channels,
                                  kernel_size=kernel_size)
            self.layers.append(cc_block)
            setattr(self, 'cc_layer{}'.format(i), cc_block)

            # Done to ensure layer information prints out when print() is called

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        return out

    def set_device(self, device):
        self.device = device