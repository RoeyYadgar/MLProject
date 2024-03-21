
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from conv_layers import ConvBlock,DownConvBlock,UpConvBlock

class Encoder(nn.Module):
    
    def __init__(self, image_size: int = 32, ch_num: int = 64, conv_blocks: int = 3, ch_size: int = 8):
        """
        Encoder network which transforms image into representation space
        
        Args:
        
            image_size (int): The size of the input images (assumed to be square).
            ch_num (int): number of output channels of the network.
            conv_blocks (int): number of convultional blocks of the network (each block consists of 2 convultional layer, batch normalization and PReLu Activation, and a downsample if necessary) 
            ch_size (int): size of each output channel of the network.
            
        """
        super().__init__()
        
        
        ch_in = 1
        ch_out = ch_num // (2**(conv_blocks-1))
        
        curr_ch_size = 32
    
        self.conv_layers = nn.ModuleList()
        
        for i in range(conv_blocks):
            if(curr_ch_size == ch_size): #Use DownConvBlock which downsamples by a factor of 2 until the channel size gets to the desired ch_size
                self.conv_layers.append(ConvBlock(ch_in, ch_out, 3))
            else:
                self.conv_layers.append(DownConvBlock(ch_in, ch_out, 3))
            ch_in = ch_out
            ch_out *=2
            curr_ch_size//=2
           
         
        batch_size = 32
        #Use normalization layer and Relu layer as the last layer of the encoder
        self.norm_layer = nn.LayerNorm((ch_num,ch_size,ch_size),eps=0,elementwise_affine=False)
        self.Relu_layer = nn.ReLU()
        
        # assign parameters
        self.ch_num = ch_num
        self.ch_size = ch_size
        self.image_size = image_size
        self.conv_blocks = conv_blocks
        
    def forward(self,x):
        for layer in self.conv_layers:
            x = layer(x)
            
        x = self.norm_layer(x)
        x = self.Relu_layer(x)
        return x
    
    
class Decoder(nn.Module):
    def __init__(self,image_size: int = 32, ch_num: int = 64, conv_blocks: int = 3, ch_size: int = 8):
        """
        Decoder network which transforms representation space back into image
        
        Args:
        
            image_size (int): The size of the input images (assumed to be square).
            ch_num (int): number of intput channels of the network.
            conv_blocks (int): number of convultional blocks of the network (each block consists of 2 convultional layer, batch normalization and PReLu Activation, and a downsample if necessary) 
            ch_size (int): size of each input channel of the network.
            
        """
        super().__init__()
        
        
        ch_in = ch_num
        ch_out = int(ch_num / 2)
    
        self.conv_layers = nn.ModuleList()
        
        nonUpConvBlocks = conv_blocks - int(np.log2(image_size/ch_size))
        
        for i in range(conv_blocks):
            if(i == conv_blocks-1):#Use UpConvBlock which upsamples by a factor of 2 until the channel size gets to the image_size
                ch_out = 1
            if(i < nonUpConvBlocks):
                self.conv_layers.append(ConvBlock(ch_in, ch_out, 3))
            else:
                self.conv_layers.append(UpConvBlock(ch_in, ch_out, 3))
            ch_in = ch_out
            ch_out = int(ch_out/2)
            
            
        self.ch_num = ch_num
        self.ch_size = ch_size
        self.image_size = image_size
        self.conv_blocks = conv_blocks
           
        
    def forward(self,x):
        for layer in self.conv_layers:
            x = layer(x)
        return x
    
class EncoderDecoderNet(nn.Module):
    def __init__(self,image_size: int = 32, ch_num: int = 64, conv_blocks: int = 3,l1_loss_lambda: int = 1):
        """
        Encoder-Decoder network which transforms image to represenation space and back into image, used later in DecoderArchNet
        
        Args:
        
            image_size (int): The size of the input images (assumed to be square).
            ch_num (int): number of channels of the encoder and decoder networks.
            conv_blocks (int): number of convultional blocks of the network (each block consists of 2 convultional layer, batch normalization and PReLu Activation, and a downsample if necessary) 
            l1_loss_lambda (int): weight of l1 representation norm loss
        """
        super(EncoderDecoderNet,self).__init__()
        self.encoder = Encoder(image_size,ch_num,conv_blocks)
        self.decoder = Decoder(image_size,ch_num,conv_blocks)
        
        self.l1_loss_lambda = l1_loss_lambda
        
    def forward(self,x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def encode(self,x):
        x = x.view(x.size(0),1,x.size(1),x.size(2))
        return self.encoder(x)
    
    def decode(self,x):
        x = self.decoder(x)
        x = x.view(x.size(0),x.size(2),x.size(3))
        return x
                
    def compute_loss(self,data,target):
         encoded_data = self.encode(target)
         decoded_data = self.decode(encoded_data)
         
         loss = nn.MSELoss()(decoded_data, target) + self.l1_loss_lambda * torch.norm(encoded_data,p=1)             
         return loss
     
    def predict(self,x):
         return self(x)
              
    
        
        