import torch.nn as nn

import model.layers.fusion_net_layers as layers


class FusionNet(nn.Module):

    def __init__(self, args, dims):
        """FucisionNet: Residual convolutional network for segmentaion

        Parameters:
        ----------
            args: ArgumentParser()
                hyperparamerter argument parsers
            dims: int
                dimention of the input data

        References:
        ----------
            https://arxiv.org/abs/1612.05360
        """

        super(FusionNet, self).__init__()
        num_kernel = args.num_kernel
        kernel_size = args.kernel_size

        self.conv_down_1 = layers.ConvLayer(dims, num_kernel, kernel_size)
        self.res_down_1 = layers.ResidualBlock(num_kernel, kernel_size)
        self.down_sample_1 = layers.DownSampling(num_kernel, kernel_size)

        self.conv_down_2 = layers.ConvLayer(num_kernel, num_kernel*2, kernel_size)
        self.res_down_2 = layers.ResidualBlock(num_kernel*2, kernel_size)
        self.down_sample_2 = layers.DownSampling(num_kernel*2, kernel_size)

        self.conv_down_3 = layers.ConvLayer(num_kernel*2, num_kernel*4, kernel_size)
        self.res_down_3 = layers.ResidualBlock(num_kernel*4, kernel_size)
        self.down_sample_3 = layers.DownSampling(num_kernel*4, kernel_size)

        self.conv_down_4 = layers.ConvLayer(num_kernel*4, num_kernel*8, kernel_size)
        self.res_down_4 = layers.ResidualBlock(num_kernel*8, kernel_size)
        self.down_sample_4 = layers.DownSampling(num_kernel*8, kernel_size)

        self.conv_mid = layers.ConvLayer(num_kernel*8, num_kernel*16, kernel_size)
        self.res_mid = layers.ResidualBlock(num_kernel*16, kernel_size)
        self.up_sample_mid = layers.UpSampling(num_kernel*16, num_kernel*8, kernel_size)

        self.conv_1 = layers.ConvLayer(num_kernel*8, num_kernel*8, kernel_size)
        self.res_1 = layers.ResidualBlock(num_kernel*8, kernel_size)
        self.up_sample_1 = layers.UpSampling(num_kernel*8, num_kernel*4, kernel_size)

        self.conv_2 = layers.ConvLayer(num_kernel*4, num_kernel*4, kernel_size)
        self.res_2 = layers.ResidualBlock(num_kernel*4, kernel_size)
        self.up_sample_2 = layers.UpSampling(num_kernel*4, num_kernel*2, kernel_size)

        self.conv_3 = layers.ConvLayer(num_kernel*2, num_kernel*2, kernel_size)
        self.res_3 = layers.ResidualBlock(num_kernel*2, kernel_size)
        self.up_sample_3 = layers.UpSampling(num_kernel*2, num_kernel, kernel_size)

        self.conv_4 = layers.ConvLayer(num_kernel, num_kernel, kernel_size)
        self.res_4 = layers.ResidualBlock(num_kernel, kernel_size)
        self.conv_5 = layers.ConvLayer(num_kernel, num_kernel, kernel_size)

        #self.output = layers.ConvLayer(num_kernel, 1, kernel_size)
        self.output = nn.Conv2d(num_kernel, 1, 1, padding=0, stride=1)


    def forward(self, x):

        x = self.conv_down_1(x)
        x = self.res_down_1(x)
        x, skip1 = self.down_sample_1(x)

        x = self.conv_down_2(x)
        x = self.res_down_2(x)
        x, skip2 = self.down_sample_2(x)

        x = self.conv_down_3(x)
        x = self.res_down_3(x)
        x, skip3 = self.down_sample_3(x)

        x = self.conv_down_4(x)
        x = self.res_down_4(x)
        x, skip4 = self.down_sample_4(x)

        x = self.conv_mid(x)
        x = self.res_mid(x)
        x = self.up_sample_mid(x, skip4)

        x = self.conv_1(x)
        x = self.res_1(x)
        x = self.up_sample_1(x, skip3)

        x = self.conv_2(x)
        x = self.res_2(x)
        x = self.up_sample_2(x, skip2)

        x = self.conv_3(x)
        x = self.res_3(x)
        x = self.up_sample_3(x, skip1)

        x = self.conv_4(x)
        x = self.res_4(x)
        x = self.conv_5(x)

        pred = nn.functional.sigmoid(self.output(x))

        return pred

