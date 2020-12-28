'''
Author: sigmoid
Description: 
Email: 595495856@qq.com
Date: 2020-12-28 12:00:19
LastEditTime: 2020-12-28 14:07:49
'''
import torch
import torch.nn as nn

from models.cbam import Cbam

# encoder params
num_denseblock = 3
depth = 16
growth_rate = 24

class BottleneckBlock(nn.Module):
    """
    Dense Bottleneck Block
    It contains two convolutional layers, a 1x1 and a 3x3.
    """

    def __init__(self, input_size, growth_rate, dropout_rate=0.2):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added. That is the ouput
                size of the last convolutional layer.
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(BottleneckBlock, self).__init__()
        inter_size = 4*growth_rate

        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(inter_size)
        self.conv1 = nn.Conv2d(
            input_size, inter_size, kernel_size=1, stride=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(
            inter_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.norm1(self.relu(self.conv1(x)))
        out = self.norm2(self.relu(self.conv2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    """
    Transition Block
    A transition layer reduces the number of feature maps in-between two bottleneck
    blocks.
    """

    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the output
        """
        super(TransitionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            input_size, output_size, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.norm(self.relu(self.conv(x)))
        return self.pool(out)
 
class DenseBlock(nn.Module):
    """
    Dense block
    A dense block stacks several bottleneck blocks.
    """

    def __init__(self, input_size, growth_rate, depth, dropout_rate=0.2):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added per bottleneck block
            depth (int): Number of bottleneck blocks
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(DenseBlock, self).__init__()
        layers = [
            BottleneckBlock(
                input_size + i * growth_rate, growth_rate, dropout_rate=dropout_rate
            )
            for i in range(depth)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    """Multi-scale Dense Encoder
    A multi-scale dense encoder with two branches. The first branch produces
    low-resolution annotations, as a regular dense encoder would, and the second branch
    produces high-resolution annotations.
    """

    def __init__(
        self, img_channels=1, num_in_features=48, dropout_rate=0.2, checkpoint=None
    ):
        """
        Args:
            img_channels (int, optional): Number of channels of the images [Default: 1]
            num_in_features (int, optional): Number of channels that are created from
                the input to feed to the first dense block [Default: 48]
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
            checkpoint (dict, optional): State dictionary to be loaded
        """
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(
            img_channels,
            num_in_features,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        num_features = num_in_features
        self.block1 = DenseBlock(
            num_features,
            growth_rate=growth_rate,
            depth=depth,
            dropout_rate=dropout_rate,
        )

        num_features = num_features + depth * growth_rate
        self.cbam1 = Cbam(num_features)
        self.trans1 = TransitionBlock(num_features, num_features // 2)
        num_features = num_features // 2
        self.block2 = DenseBlock(
            num_features,
            growth_rate=growth_rate,
            depth=depth,
            dropout_rate=dropout_rate,
        )

        num_features = num_features + depth * growth_rate
        self.cbam2 = Cbam(num_features)
        self.trans2 = TransitionBlock(num_features, num_features // 2)

        num_features = num_features // 2
        self.block3 = DenseBlock(
            num_features,
            growth_rate=growth_rate,
            depth=depth,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu(self.norm0(out))
        out = self.max_pool(out)
        out = self.block1(out)
        # out = self.cbam1(out)
        out = self.trans1(out)
        out = self.block2(out)
        # out = self.cbam2(out)
        out = self.trans2(out)
        out = self.block3(out)
        return out
