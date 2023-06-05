import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
from core_qnn.quaternion_layers import *

# 改进：每个SIPNNBlock加入SEModule
# 改进：把PAN_HP的卷积增加到4
# 增加deconv，ms用于训练而不是lms
# SPITAL 从4改8,网络改成resblock
# 最后输出前直接加lms的结果
# SIPNNBlock中卷积改为2，block个数减半，先SE再加法，同时加spatial和残差
# 不使用hp，用PAN-MS训练（Detail Injection-Based Deep Convolutional Neural Networks for Pansharpening
# 不使用lms，ms由lms得到


# --------------SE Block-------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, reduction=2):
        super(SEBlock, self).__init__()

        ch_in = 64
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):  # x = ms; y =output
        b, c, _, _ = x.size()
        w = self.avg_pool(x).view(b, c)  # squeeze操作
        w = self.fc(w).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return y * w.expand_as(y)  # 注意力作用每一个通道上


# resblock
class Resblock(nn.Module):
    def __init__(self, channel=64):
        super(Resblock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x = input; y = output
        rs1 = self.relu(self.conv1(x))  # Bsx32x64x64
        rs1 = self.conv2(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs

class SEResblock(nn.Module):
    def __init__(self, channel=64, reduction=2):
        super(SEResblock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x = input; y = output
        rs1 = self.relu(self.conv1(x))  # Bsx32x64x64
        rs1 = self.conv2(rs1)  # Bsx32x64x64
        b, c, _, _ = x.size()
        w = self.avg_pool(x).view(b, c)  # squeeze操作
        w = self.fc(w).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        rs1 = rs1 * w.expand_as(rs1)
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


# class SEResblock(nn.Module):
#     def __init__(self, channel=64, reduction=2):
#         super(SEResblock, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
#                                 bias=True)
#         self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
#                                 bias=True)
#         self.se = SEBlock()
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):  # x = input; y = output
#         rs1 = self.relu(self.conv1(x))  # Bsx32x64x64
#         rs1 = self.conv2(rs1)  # Bsx32x64x64
#         rs1 = self.se(x, rs1)
#         rs = torch.add(x, rs1)  # Bsx32x64x64
#         return rs


class SIPNN_block(nn.Module):
    def __init__(self):
        super(SIPNN_block, self).__init__()
        #self.Qconv = QuaternionConv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.se = SEBlock()
        self.relu = nn.ReLU()

    def forward(self, lms, spatial_info):
        y = self.relu(self.conv1(lms))
        y = self.conv2(y)
        output = self.se(lms, y) + spatial_info + lms
        return output


# 光谱分支，改为resnet结构
class SPATIAL_Branch(nn.Module):
    def __init__(self, spectral_num=8, res_block_num=4):
        super(SPATIAL_Branch, self).__init__()
        self.spectral_num = spectral_num
        self.deconv1 = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4, padding=2, bias=True)
        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=64, kernel_size=3, padding=1, bias=True)  # 第一层卷积
        self.relu = nn.ReLU()
        # 构造branch的骨干网络
        res_backbone_list = []
        for i in range(res_block_num):
            res_backbone_list.append(Resblock())
        self.res_backbone = nn.Sequential(*res_backbone_list)

    def forward(self, lms, pan):  # x=pan_hp, y=output
        pan_d = pan.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_d, lms)
        y = self.relu(self.conv1(input))
        y = self.res_backbone(y)
        return y


class SIPNN(nn.Module):
    def __init__(self, res_block_num=4, block_num=4, spectral_num=8):
        super(SIPNN, self).__init__()
        # self.SIPNN_block = SIPNN_block()
        self.SIPNN_backbone = nn.ModuleList([SIPNN_block() for i in range(block_num)])
        self.deconv1 = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4,
                                          padding=2, bias=True)
        self.deconv2 = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4,
                                          padding=2, bias=True)
        self.conv1_lms = nn.Conv2d(in_channels=spectral_num, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.spatial = SPATIAL_Branch(spectral_num, res_block_num)
        self.conv_final = nn.Conv2d(in_channels=64, out_channels=spectral_num, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, ms, pan):
        lms = self.deconv1(ms)
        lms_feature = self.relu(self.conv1_lms(lms))
        spatial_expand = self.relu(self.spatial(lms, pan))
        for block in self.SIPNN_backbone:
            lms_feature = block(lms_feature, spatial_expand)
        output = self.conv_final(lms_feature) + self.deconv2(ms)
        return output


# 引自老代码
def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x/10*1.28

    variance_scaling(tensor)

    return tensor


def summaries(model, writer=None, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 16, 16), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model, (x,))

