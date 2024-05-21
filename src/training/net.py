from typing import List
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(
        self, conv_sizes: List[int], linear_sizes: List[int], output_size: int
    ):
        super(ConvNet, self).__init__()

        conv_block = []

        for a, b in zip(conv_sizes[:-1], conv_sizes[1:]):
            conv_block.append(nn.Conv2d(a, b, kernel_size=5))
            conv_block.append(nn.ReLU())
            conv_block.append(nn.MaxPool2d(2))

        self.conv_block = nn.Sequential(*conv_block)
        self.flatten = nn.Flatten()

        linear_block = []

        for a, b in zip(linear_sizes[:-1], linear_sizes[1:]):
            linear_block.append(nn.Linear(a, b))
            linear_block.append(nn.ReLU())

        linear_block.append(nn.Linear(linear_sizes[-1], output_size))
        linear_block.append(nn.Tanh())

        self.linear_block = nn.Sequential(*linear_block)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.linear_block(x)
        return x
