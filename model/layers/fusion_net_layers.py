"""Building blocks for the FusionNet model
"""

import torch.nn as nn


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1,name=None):

        super(ConvLayer, self).__init__()

        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_channels))

        self.conv_layer = nn.Sequential(*block)


    def forward(self, x):
        
        output = self.conv_layer(x)

        return output


class ResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size, name=None):

        super(ResidualBlock, self).__init__()
        block = [ConvLayer(channels, channels, kernel_size) for _ in range(3)]
        self.residual_block = nn.Sequential(*block)


    def forward(self, x):
        
        output = self.residual_block(x)

        output += x

        return output


class DownSampling(nn.Module):

    def __init__(self, channels, kernel_size, name=None):
        super(DownSampling, self).__init__()

        self.conv = ConvLayer(channels, channels, kernel_size)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        
        conv_out = self.conv(x) 
        output = self.max_pool(conv_out)

        return output, conv_out



class UpSampling(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, name=None):
        super(UpSampling, self).__init__()

        self.conv = ConvLayer(in_channels, out_channels, kernel_size)
        self.conv_t = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, padding=1, stride=2, output_padding=1) 


    def forward(self, x, skip):
        
        conv_out = self.conv(x) 
        output = self.conv_t(conv_out)

        output += skip

        return output 
