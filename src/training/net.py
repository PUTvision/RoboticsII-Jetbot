from typing import List
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(
        self, conv_sizes: List[int], linear_sizes: List[int], output_size: int
    ):
        super(ConvNet, self).__init__()

        blocks = []

        for a, b in zip(conv_sizes[:-1], conv_sizes[1:]):
            blocks.append(nn.Conv2d(a, b, kernel_size=5))
            blocks.append(nn.ReLU())
            blocks.append(nn.BatchNorm2d(b))
            blocks.append(nn.MaxPool2d(2))

        blocks.append(nn.Flatten())

        for a, b in zip(linear_sizes[:-1], linear_sizes[1:]):
            blocks.append(nn.Linear(a, b))
            blocks.append(nn.ReLU())

        blocks.append(nn.Linear(linear_sizes[-1], output_size))
        blocks.append(nn.Tanh())

        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)
