# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 20:00:35 2023

@author: User
"""


import pickle
import torch
import numpy as np
import torch.nn as nn


class waveletLayer(nn.Module):
    
    def __init__(self,inverse=False):
        
        super(waveletLayer,self).__init__()
        
        wavelet_weights_path = 'wavelet_haar_weights_c2.pkl'
        with open(wavelet_weights_path,'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            wavelet_weights = u.load()
            
        kernel_size = 8
        subband_channels = kernel_size ** 2
        wavelet_weights = torch.from_numpy(wavelet_weights['rec%d' % kernel_size])[:subband_channels]
        
        
        if(not inverse):
            self.wavelet_transform = nn.Conv2d(1,subband_channels,kernel_size,stride = kernel_size, padding=0,groups=1,padding_mode='zeros',bias=False)
        else:
            self.wavelet_transform = nn.ConvTranspose2d(subband_channels,1,kernel_size,stride = kernel_size, padding=0,groups=1,padding_mode='zeros',bias=False)
            
        self.wavelet_transform.weight.data = wavelet_weights
        self.wavelet_transform.weight.bias = torch.zeros(subband_channels)
        self.wavelet_transform.weight.requires_grad = False
        
    def forward(self,x):
        output = self.wavelet_transform(x)
        return output
    
    
def convetWaveletChannels2img(x):
    #Converts a tensor of wavelet space of size(num_img,wavelet_channels,kernel_size,kernel_size)
    #to a tensor of size(num_img,img_len,img_len) 
    #which contains the same wavelet transform values but can be display using plt.imshow(x[i])
    num_channels = x.size(1)
    sqrt_num_channels = int(np.sqrt(num_channels))
    x = x.view(x.size(0),sqrt_num_channels,sqrt_num_channels,x.size(2),x.size(3))
    x = x.permute(0,1,3,2,4)
    img_len = int(np.sqrt(x.numel()/x.size(0)))
    x = x.reshape(x.size(0),img_len,img_len)
    
    return x