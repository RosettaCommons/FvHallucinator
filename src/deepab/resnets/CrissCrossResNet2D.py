"""Criss-cross attention based on implementation from:
https://github.com/Serge-weihao/CCNet-Pure-Pytorch

MIT License

Copyright (c) 2019 Serge-weihao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.deepab.resnets.ResNet2D import ResBlock2D


class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim // 8,
                                    kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim,
                                  out_channels=in_dim // 8,
                                  kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim,
                                    kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = lambda B, H, W: -torch.diag(
            torch.tensor(float("inf")).to(self.device).repeat(H), 0).unsqueeze(
                0).repeat(B * W, 1, 1)
        # self.INF = lambda B, H, W, device: -torch.diag(
        #     torch.tensor(float("inf")).to(device).repeat(H), 0).unsqueeze(
        #         0).repeat(B * W, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_type)

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(
            m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(
            m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(
            m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(
            m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(
            m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(
            m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) +
                    self.INF(m_batchsize, height, width)).view(
                        m_batchsize, width, height,
                        height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W,
                             proj_key_W).view(m_batchsize, height, width,
                                              width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :,
                        0:height].permute(0, 2, 1, 3).contiguous().view(
                            m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(
            m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H,
                          att_H.permute(0, 2,
                                        1)).view(m_batchsize, width, -1,
                                                 height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W,
                          att_W.permute(0, 2,
                                        1)).view(m_batchsize, height, -1,
                                                 width).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x, concate


class RCCAModule(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 return_attn=False,
                 norm=nn.BatchNorm2d):
        super(RCCAModule, self).__init__()
        self.return_attn = return_attn
        inter_channels = in_channels // 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      inter_channels,
                      kernel_size=(kernel_size, kernel_size),
                      stride=(1, 1),
                      padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2),
                      bias=False), norm(inter_channels), nn.ReLU())
        self.cca = CrissCrossAttention(inter_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channels,
                      in_channels,
                      kernel_size=(kernel_size, kernel_size),
                      stride=(1, 1),
                      padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2),
                      bias=False), norm(in_channels), nn.ReLU())

    def forward(self, x):
        output = self.conv1(x)
        attns = []
        for _ in range(2):
            output, attn = checkpoint(self.cca, output)
            attns.append(attn)
        output = self.conv2(output)

        if self.return_attn:
            return output, attns
        else:
            return output


class CCResNet2D(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 num_blocks,
                 cc_frequency=1,
                 kernel_size=3,
                 dilation_cycle=5,
                 norm=nn.BatchNorm2d):

        super(CCResNet2D, self).__init__()

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
        self.bn1 = norm(self.channels)

        self.layers = []
        for i in range(self.num_blocks):
            dilation = int(math.pow(
                2, i % dilation_cycle)) if dilation_cycle > 0 else 1
            block = ResBlock2D(in_channels=self.channels,
                               kernel_size=kernel_size,
                               dilation=dilation)
            self.layers.append(block)
            setattr(self, 'layer{}'.format(i), block)

            if (i + 1) % cc_frequency == 0:
                cc_block = RCCAModule(in_channels=self.channels,
                                      kernel_size=kernel_size,
                                      norm=norm)
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
