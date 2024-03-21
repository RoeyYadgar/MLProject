import torch
import torch.nn as nn
from typing import List, Union, Tuple

from conv_layers import ConvBlock
from losses import MSEbatchloss
from typing import List

class DecoderArchNet(nn.Module):
    def __init__(self, image_size: int, fully_connected_layers: List[int], conv_layers: List[int],
                 encoder_net: nn.Module, decoder_net: nn.Module,
                 rep_loss_lambda: float = 1, rot180: bool = True):
        """
        A PyTorch descendent module that implements a decoder architecture network.

        Args:
            image_size (int): The size of the input images (assumed to be square).
            fully_connected_layers (list[int]): A list of layer sizes for the fully connected layers.
            conv_layers (list[ConvBlock]): A list of convolutional layers to apply to the input.
            encoder_net (nn.Module): The encoder network to use for computing the representation loss.
            decoder_net (nn.Module): The decoder network to use for reconstructing the input.
            rep_loss_lambda (float, optional): The weight to use for the representation loss. Defaults to 1.
            rot180 (bool, optional): Whether to apply a 180-degree rotation to the input. Defaults to True.

        Returns:
            None
        """
        super().__init__()
        
        self.image_size = image_size
        self.fc_layers = nn.ModuleList()
        layer_input_size = image_size**2
        
        # add fully connected layers with PReLU activation
        for layer_size in fully_connected_layers:
            self.fc_layers.append(nn.Linear(layer_input_size,layer_size))
            self.fc_layers.append(nn.PReLU())
            layer_input_size = layer_size
            
        # add encoders and decoders
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        
        # pre-trained
        for param in self.encoder_net.parameters():
            param.requires_grad = False
        
        for param in self.decoder_net.parameters():
            param.requires_grad = False
        
        
        self.channel_size = int(self.encoder_net.ch_size)
        self.channel_num = self.encoder_net.ch_num
        
        # add conv layers
        self.conv_layers = nn.ModuleList()
        channel_input_size = self.channel_num
        for channel_size in conv_layers:
            self.conv_layers.append(ConvBlock(channel_input_size, channel_size, 3))
            channel_input_size = channel_size
        
        # additional parameters
        self.rep_loss_lambda = rep_loss_lambda
        self.rot180 = rot180
          
    def forward(self, x: torch.Tensor, return_intermediate: bool =False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        
        forward pass through the network, option to return intermediate layer (without decoding)
        
        Args:
            x (torch.Tensor): input tensor
            return_intermediate (bool, optional): should return the intermediate layer. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: output tensor or tuple of output tensor and intermediate layer
        """
        x = x.view(x.size(0), -1)
        
        # forward through fully connected
        for layer in self.fc_layers:
            x = layer(x)

        # reshape to multi channel
        x = x.view(x.size(0), self.channel_num, self.channel_size,self.channel_size)
        
        # forward through conv layers
        for layer in self.conv_layers:
            x = layer(x)
        
        intermediate_layer = x
        
        # decode and reshape
        x = self.decoder_net(x)
        x = x.view(x.size(0), self.image_size, self.image_size)
        
        if(return_intermediate):
            return x, intermediate_layer
        else:
            return x
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """predict"""
        prediction = self(x)
        return prediction
    
    def compute_loss(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        compute loss for batch

        Args:
            data (torch.Tensor): input tensor
            target (torch.Tensor): target tensor

        Returns:
            torch.Tensor: loss tensor
        """
        output, intermediate_layer = self(data, return_intermediate=True)
        encoded_target = self.encoder_net(target.view(target.size(0), 1, target.size(1), target.size(2)))
        batch_loss = MSEbatchloss(output, target) + self.rep_loss_lambda * MSEbatchloss(intermediate_layer, encoded_target)
        
        # compute loss for rotated image
        if(self.rot180):
            rotated_target = torch.rot90(target, 2 ,(-2,-1))
            rotated_encoded_target = self.encoder_net(rotated_target.view(target.size(0),1,target.size(1),target.size(2)))
            rotated_batch_loss = MSEbatchloss(output, rotated_target) + self.rep_loss_lambda * MSEbatchloss(intermediate_layer, rotated_encoded_target)
            
            batch_loss = torch.min(batch_loss, rotated_batch_loss)
        
        loss = torch.mean(batch_loss)
        
        return loss

    def train(self, mode: bool=True):
        """train"""
        super().train(mode)
        self.encoder_net.eval() #Keep encoder and decoder nets always on eval mode
        self.decoder_net.eval()
        return self
    
    
