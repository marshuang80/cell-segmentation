import torch.nn as nn

class DilationCNN(nn.Module):

    def __init__(self, dims):
        """A Dilated Convolutional Neural Network Model
            
        Parameters
        ----------
            dims: int
                number of dimentions for the input data
        """

        super(DilationCNN, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels=dims,\
                                out_channels=8,\
                                kernel_size=3,
                                padding=1,\
                                stride=1))
        layers.append(nn.Conv2d(in_channels=8,\
                                out_channels=8,\
                                kernel_size=3,
                                padding=1,\
                                stride=1))
        layers.append(nn.Conv2d(in_channels=8,\
                                out_channels=16,\
                                dilation=2, \
                                kernel_size=3,
                                padding=2,\
                                stride=1))
        layers.append(nn.Conv2d(in_channels=16,\
                                out_channels=16,\
                                dilation=2, \
                                kernel_size=3,
                                padding=2,\
                                stride=1))
        layers.append(nn.Conv2d(in_channels=16,\
                                out_channels=32,\
                                dilation=3, \
                                kernel_size=3,
                                padding=3,\
                                stride=1))
        layers.append(nn.Conv2d(in_channels=32,\
                                out_channels=16,\
                                kernel_size=1,
                                padding=0,\
                                stride=1))
        layers.append(nn.Conv2d(in_channels=16,\
                                out_channels=1,\
                                kernel_size=1,
                                padding=0,\
                                stride=1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        output = self.net(x)

        return output

