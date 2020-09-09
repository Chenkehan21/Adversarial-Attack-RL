import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        """output shape (nsteps,84,84)"""
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            # in_channels = input_shape[0] = nsteps
            # out_channels = 32
            # kernel_size = 8*8
            # output shape(32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # in_channels = 32
            # out_channels = 64
            # kernel_size = 4*4
            # output shape(64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # in_channels = input_shape[0] = 64
            # out_channels = 64
            # kernel_size = 3 * 3
            # output shape(64, 7, 7)  (x - kernel_size / stride) + 1
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # this can be done once on model creation

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        # torch.zeros(1, *shape) 创建一个 例子 运行一遍 conv
        # *shape 是吧shape 这个tuple 的值拆开和前面的 1 构成一个新的tuple
        return int(np.prod(o.size()))
        # o.size()是一个 tuple 可以用 np
        # np.prod 计算乘积 1 * kernels * w * h


    def forward(self, x):
        """input shape (batch, channel, x, y), 默认是从axis = 1开始执行的"""
        conv_out = self.conv(x).view(x.size()[0], -1)
        # 这里是输入的是一个torch的 tensor 所以用size 第一个维度是 batch
        return self.fc(conv_out)