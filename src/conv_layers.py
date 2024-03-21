import numpy as np
import torch.nn as nn

# definition of convultion block class and descedents with specific adaptions.

class ConvBlock(nn.Module):

    def __init__(self, ch_in: int, ch_out: int, kernel_size: int):
        """
        convolution block consisting of 2 conv layers with batch normalization and PReLU activation each.

        Args:
            ch_in (int): number of input channels
            ch_out (int): number of output channels
            kernel_size (int): kernel size
        """
        
        super().__init__()
        padding = kernel_size//2
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=kernel_size,padding=padding,stride=1),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(),
            nn.Conv2d(ch_out,ch_out,kernel_size=kernel_size,padding=padding,stride=1),
            nn.BatchNorm2d(ch_out),
            nn.PReLU()
            )
        
    def forward(self,x):
        x = self.conv_block(x)
        return x
            
class DownConvBlock(nn.Module):

    def __init__(self, ch_in: int, ch_out: int, kernel_size: int):
        """
        downsamples the input with 2 strides using average pooling after a conv block.

        Args:
            ch_in (int): number of input channels
            ch_out (int): number of output channels
            kernel_size (int): kernel size
        """
        super().__init__()
        
        # conv block with average pooling
        self.down_conv_block = nn.Sequential(
            ConvBlock(ch_in, ch_out, kernel_size),
            nn.AvgPool2d(kernel_size=2, stride=2)
            )
        
    def forward(self,x):
        x = self.down_conv_block(x)
        return x
            
class UpConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, kernel_size: int):
        """
        upsamples the input with bilinear upsampling after a conv block.
        
        Args:
            ch_in (int): number of input channels
            ch_out (int): number of output channels
            kernel_size (int): kernel size
        """
        super(UpConvBlock, self).__init__()
        
        # conv block with bilinear up-sampling
        self.up_conv_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) ,
            ConvBlock(ch_in, ch_out, kernel_size)
            )
        
    def forward(self,x):
        x = self.up_conv_block(x)
        return x