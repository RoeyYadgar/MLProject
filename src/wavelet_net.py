import pickle
import torch
import numpy as np
import torch.nn as nn
# from conv_layers import ConvBlock
from pytorch_wavelets import DWTForward, DWTInverse
from typing import List

from decoder_arch_net import DecoderArchNet

class waveletLayer(nn.Module):
    
    def __init__(self,image_size: int,inverse: bool = False,kernel_size: int = 8):
        """
        
        Wavelet layer which transforms image into Haar Wavelet Packet Space (and invert back to the image if setting inverse = True)
        
        Args:
        
            image_size (int): The size of the input images (assumed to be square).
            inverse (bool): determines if layer is the Wavelet Transform or its' inverse.Default to False
            kernel_size (int): the size of each channel of the wavelet transform (for example kernel_size = 8 means each channel is of size 8x8 and the total number of channels is image_size^2/kernel_size^2).Defaults to 8
            
        """
        super(waveletLayer, self).__init__()
        
        # load wavelet weights from file
        wavelet_weights_path = 'wavelet_haar_weights_c2.pkl'
        with open(wavelet_weights_path,'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            wavelet_weights = u.load()
            

        subband_channels = kernel_size ** 2
        wavelet_weights = torch.from_numpy(wavelet_weights['rec%d' % kernel_size])[:subband_channels] #Pick the wavelet weights corresponding to the given kernel_size
        
        
        if(not inverse):
            self.wavelet_transform = nn.Conv2d(1, subband_channels,kernel_size,stride = kernel_size, padding=0,groups=1,padding_mode='zeros',bias=False) #packet wavelet transform can be implemented by a convolutional layer with no bias component
        else:
            self.wavelet_transform = nn.ConvTranspose2d(subband_channels,1,kernel_size,stride = kernel_size, padding=0,groups=1,padding_mode='zeros',bias=False) #the inverse transform can be implemented by a transposed convoltuional layer (since the transform is orthogonal)
        
        #set the wavelet weights in the convolutional layer and use requires_grad = False to 'fix' the weights (i.e. the weights will not be trained)
        self.wavelet_transform.weight.data = wavelet_weights
        self.wavelet_transform.weight.bias = torch.zeros(subband_channels)
        self.wavelet_transform.weight.requires_grad = False
        
        self.ch_num = subband_channels
        self.ch_size = int(image_size/kernel_size)
        self.image_size = image_size
        
    def forward(self,x):
        output = self.wavelet_transform(x)
        return output
    
class DWTLayer(nn.Module):
    
    def __init__(self,image_size: int,J: int = 3):
    
        """
        Wavelet layer which transforms image into Haar Wavelet Space (not packet transform)

        Args:
            image_size (int): The size of the input images (assumed to be square).
            J (int): wavelet transform depth, default = 3
        """ 
        super(DWTLayer,self).__init__()
        self.J = J
        self.wavelet_transform = DWTForward(J, mode='zero', wave='haar')
        self.ch_size = int(image_size/(2**J))
        self.ch_num = int(image_size/self.ch_size)**2
        self.image_size = image_size
    
    def forward(self,x):
        x = x.view(x.size(0),1,self.image_size,self.image_size)
        x = self.wavelet_transform(x)
        x = convertDWT2img(x)
        x = x.view(x.size(0),self.ch_num,self.ch_size,self.ch_size)
        return x
    
class DWTInvLayer(nn.Module):
    
    def __init__(self,image_size: int,J: int = 3):
        """
        Wavelet layer which transforms Haar Wavelet Space (not packet transform) back into image

        Args:
            image_size (int): The size of the input images (assumed to be square).
            J (int): wavelet transform depth. Defaults to 3.
        """ 
        super(DWTInvLayer,self).__init__()
        self.J = J
        self.wavelet_transform = DWTInverse(mode='zero', wave='haar')
        self.ch_size = int(image_size/(2**J))
        self.ch_num = int(image_size/self.ch_size)**2
        self.image_size = image_size
        
    def forward(self,x):
        x = x.view(x.size(0),self.image_size,self.image_size)
        x = convertimg2DWT(x, self.J)
        x = self.wavelet_transform(x)
        x = x.view(x.size(0),x.size(2),x.size(3))
    
        return x
    

class waveletNet(DecoderArchNet):
    def __init__(self,image_size: int, fully_connected_layers: List[int], conv_layers: List[int],rep_loss_lambda: float = 1, rot180: bool = True,packet_transform: bool = True):
        """
        Phase Retrieval Network with wavelet transform as a decoder

        Args:
            image_size (int): The size of the input images (assumed to be square).
            fully_connected_layers (list[int]): A list of layer sizes for the fully connected layers.
            conv_layers (list[ConvBlock]): A list of convolutional layers to apply to the input.
            rep_loss_lambda (float, optional): The weight to use for the representation loss. Defaults to 1.
            rot180 (bool, optional): Whether to apply a 180-degree rotation to the input. Defaults to True.
            packet_transform (bool,optional) : Wether to use wavelet packet transform or regular wavelet transform. Defaults to True.
        """ 
        if(packet_transform):
            wavelet_layer_inverse = waveletLayer(image_size,True)
            wavelet_layer = waveletLayer(image_size,False)
        else:
            wavelet_layer_inverse = DWTInvLayer(image_size,J=3)
            wavelet_layer = DWTLayer(image_size,J=3)
            
        #waveletNet is a subclass of DecoderArchNet which is a Network for the Phase Retrieval Problem which arbritray Encoder-Decoder Pair
        super().__init__(image_size,fully_connected_layers,conv_layers,wavelet_layer,wavelet_layer_inverse,rep_loss_lambda,rot180)
    
    

def convertDWT2img(x : torch.Tensor) -> torch.Tensor:
    """
        Converts output of DWTLayer (according to https://pytorch-wavelets.readthedocs.io/en/latest/dwt.html) to 2d tensor  

        Args:
            x (torch.Tensor): tensor input


        Returns:
            torch.Tensor: tensor output
    """ 
    batch_size = x[0].size(0)
    feature_size = int(x[1][0].size(-1)*2)
    J = len(x[1])
    output = x[0].new(batch_size,feature_size,feature_size) #x.new initalizes the tensor on the same device as x
    
    for i in range(J):
        feature_size= int(feature_size/2)
        output[:,feature_size:(2*feature_size),0:feature_size] = x[1][i][:,:,0].view(batch_size,feature_size,feature_size)
        output[:,0:feature_size,feature_size:(2*feature_size)] = x[1][i][:,:,1].view(batch_size,feature_size,feature_size)
        output[:,feature_size:(2*feature_size),feature_size:(2*feature_size)] = x[1][i][:,:,2].view(batch_size,feature_size,feature_size)
    
    output[:,0:feature_size,0:feature_size] = x[0].view(batch_size,feature_size,feature_size)

    return output
    
def convertimg2DWT(x : torch.Tensor,J : int):
    """
        Converts 2d tensor to input of DWTInvLayer (according to https://pytorch-wavelets.readthedocs.io/en/latest/dwt.html)

        Args:
            x (torch.Tensor): tensor input
            J (int): wavelet transform depth

        Returns:
            torch.Tensor: tensor output
    """
    batch_size = x.size(0)
    feature_size = int(x.size(-1)/2)
    
    min_feature_size = int(feature_size / (2**(J-1)))
    x_LL = x[:,0:min_feature_size,0:min_feature_size].view(batch_size,1,min_feature_size,min_feature_size)
    
    x_H = []
    
    for i in range(J):
        x_Hj = x.new(batch_size,1,3,feature_size,feature_size) #x.new initalizes the tensor on the same device as x
        x_Hj[:,:,0] = x[:,feature_size:(2*feature_size),0:feature_size].view(batch_size,1,feature_size,feature_size)
        x_Hj[:,:,1] = x[:,0:feature_size,feature_size:(2*feature_size)].view(batch_size,1,feature_size,feature_size)
        x_Hj[:,:,2] = x[:,feature_size:(2*feature_size),feature_size:(2*feature_size)].view(batch_size,1,feature_size,feature_size)
        
        feature_size= int(feature_size/2)


        x_H.append(x_Hj)
    
    return (x_LL,x_H)
    
def convretWaveletChannels2img(x : torch.Tensor) -> torch.Tensor:
     """
        Converts a tensor of wavelet packet channels of size(num_img,wavelet_channels,kernel_size,kernel_size) to a tensor of size(num_img,img_len,img_len) 
        
        Args:
            x (torch.Tensor): tensor input

        Returns:
            torch.Tensor: tensor output
     """
     num_channels = x.size(1)
     sqrt_num_channels = int(np.sqrt(num_channels))
     x = x.view(x.size(0),sqrt_num_channels,sqrt_num_channels,x.size(2),x.size(3))
     x = x.permute(0,1,3,2,4)
     img_len = int(np.sqrt(x.numel()/x.size(0)))
     x = x.reshape(x.size(0),img_len,img_len)
   
     return x