
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import torch.nn as nn
from torchvision import datasets,transforms
from torchvision.transforms import InterpolationMode
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.fft import fftshift,fft2
import random
from conv_layers import *
from losses import *
from typing import List

class dataSet(Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        """

        data container consisting of data and targets tensors.

        Args:
            data (torch.Tensor): input data
            targets (torch.Tensor): target outputs
        """
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_item = self.data[index]
        target_item = self.targets[index]
        return data_item, target_item
    

class Net(nn.Module):
    def __init__(self,image_size: int, fully_connected_layers: List[int], conv_layers: List[int],loss = MSEloss):
        """
        Basic Phase Retrieval Network which consists only of fully connected layers and conv layers.

        Args:
            image_size (int): The size of the input images (assumed to be square).
            fully_connected_layers (list[int]): A list of layer sizes for the fully connected layers.
            conv_layers (list[ConvBlock]): A list of convolutional layers to apply to the input.
            loss : loss function to be used for training
        """ 
        super(Net,self).__init__()
        
        self.image_size = image_size
        self.fc_layers = nn.ModuleList()
        layer_input_size = image_size**2
        for layer_size in fully_connected_layers:
            self.fc_layers.append(nn.Linear(layer_input_size,layer_size))
            self.fc_layers.append(nn.PReLU())
            layer_input_size = layer_size
            
            
        self.conv_layers = nn.ModuleList()
        channel_input_size = 1
        for channel_size in conv_layers:
            self.conv_layers.append(ConvBlock(channel_input_size, channel_size, 3))
            channel_input_size = channel_size
        
        self.loss = loss
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        pass input tensor through the network 

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = x.view(x.size(0), -1)
        
        # forward through fully connected
        for layer in self.fc_layers:
            x = layer(x)
        
        # one channel 2d image tensor
        x = x.view(x.size(0), 1, self.image_size, self.image_size)
        
        # conv layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # 2d image tensor
        x = x.view(x.size(0), self.image_size, self.image_size)
        
        return x
    
    def compute_loss(self,data,target):
        output = self(data)
        return self.loss(output,target)

    

def load_data(data,resize_len = 28,batch_size = 32 ,random_transform_enabled = True):
    """
        Performs Pre-processing of the data and returns a DataLoader 

        Args:
            data (torch.Tensor): input data of original images
            resize_len (int): image size after resize is performed to the original images
            batch_size (int): batch size of data loader
            random_transform_enabled (bool) : wether to apply augmentation the data

        Returns:
            DataLoader : DataLoader object which contains the fourier magnitudes as input and original image as target
    """
    data = data.data.float()/255
    
    # normalize as in paper
    transform = transforms.Normalize(0.1307,0.3081)
    data = transform(data)
    
    # resize if needed
    if(resize_len != 28):
        scalingTransform = transforms.Resize(resize_len,antialias='True')
        data = scalingTransform(data)
    
    if random_transform_enabled:
        p = 0.5
        translation_fraction = 0.3 # multipled by the image heigh / width
        random_transform = transforms.RandomApply([transforms.RandomAffine(degrees=0
                                                ,translate = (translation_fraction, translation_fraction)
                                                ,scale=(0.9,1.2)
                                                ,interpolation=InterpolationMode.BILINEAR)]
                                                ,p = p)
                                                
        data = random_transform(data)
    
    
    data_fft = torch.abs(fftshift(fft2(data),dim=(1,2)))
    
    dataset = dataSet(data_fft, data)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader
    
def load_MNIST(data_type, resize_len = 28, batch_size = 32, random_transform_enabled = True):
    """
        Loads MNIST Data 

        Args:
            data_type (string): 'train' or 'test' data set.
            resize_len (int): image size after resize is performed to the original images
            batch_size (int): batch size of data loader
            random_transform_enabled (bool) : wether to apply augmentation the data

        Returns:
            DataLoader : DataLoader object which contains the fourier magnitudes as input and original image as target
    """
    if(data_type == 'train'):
        data = datasets.MNIST("",download = True,train=True)
    elif(data_type == 'test'):
        data = datasets.MNIST("",download = True,train=False)
    
    return load_data(data,resize_len,batch_size,random_transform_enabled)
    


def load_FMNIST(data_type, resize_len = 28, batch_size = 32, random_transform_enabled = True):
    """
        Loads Fashion-MNIST Data 

        Args:
            data_type (string): 'train' or 'test' data set.
            resize_len (int): image size after resize is performed to the original images
            batch_size (int): batch size of data loader
            random_transform_enabled (bool) : wether to apply augmentation the data

        Returns:
            DataLoader : DataLoader object which contains the fourier magnitudes as input and original image as target
    """
    if(data_type == 'train'):
        data = datasets.FashionMNIST("",download = True,train=True)
    elif(data_type == 'test'):
        data = datasets.FashionMNIST("",download = True,train=False)
    
    return load_data(data, resize_len,batch_size,random_transform_enabled)


def load_DATASET(dataset_name, data_type, resize_len = 28, batch_size = 32, random_transform_enabled = True):
    """
        Loads Dataset

        Args:
            dataset_name (string): 'MNIST' or 'Fashion-MNIST' data set
            data_type (string): 'train' or 'test' data set
            resize_len (int): image size after resize is performed to the original images
            batch_size (int): batch size of data loader
            random_transform_enabled (bool) : wether to apply augmentation the data

        Returns:
            DataLoader : DataLoader object which contains the fourier magnitudes as input and original image as target
    """
    if(dataset_name == 'MNIST'):
        dataset_function = datasets.MNIST
    elif(dataset_name == 'Fashion-MNIST'):
        dataset_function = datasets.FashionMNIST
        
        
    train_val = True if data_type == 'train' else False
    data = dataset_function("",download=True,train=train_val)
    
    return load_data(data,resize_len,batch_size,random_transform_enabled)

def randomMisalignedData(data_loader,translation_fraction = 0.5):    
    """
        performs mis-alignment of images by performing random circular shift on the input

        Args:
            data_loader (DataLoader): data loader which contains the original images
            translation_fraction (float) : maximal fraction of the image to be shifted

        Returns:
            DataLoader : new DataLoader object after transforming images
    """
    
    target = data_loader.dataset.targets
    for i in range(target.size(0)):
        shift_h = int((random.random()*2-1)*translation_fraction*target.size(1))
        shift_v = int((random.random()*2-1)*translation_fraction*target.size(2))
        target[i] = torch.roll(target[i],(shift_h,shift_v),dims=(0,1))
    data_loader.dataset.targets = target
    
    data_loader.dataset.data = torch.abs(fftshift(fft2(data_loader.dataset.targets),dim=(1,2)))
    
    return data_loader


def saveModel(model,model_name,data_name = 'MNIST'):
    """
        Saves model to file under Models/data_name folder

        Args:
            model (nn.Module): trained model to be saved
            model_name (string) : name of model
            data_name (string) : name of dataset ('MNIST' or 'Fashion-MNIST')

      
    """
    model.to('cpu')
    if(not('.pt' in model_name)):
        model_name +='.pt'
    torch.save(model,join('Models',data_name,model_name))

def loadModel(model_name,device = 'cpu',data_name = 'MNIST'):
    """
        Loads model from file under Models/data_name folder

        Args:
            model_name (string) : name of model
            device (string) : device to load model to
            data_name (string) : name of dataset ('MNIST' or 'Fashion-MNIST')

        Returns: 
            nn.Module :  loaded model
    """
    if(not('.pt' in model_name)):
        model_name +='.pt'
    model = torch.load(join('Models',data_name,model_name))
    model.to(device)
    return model
    
if __name__ == "__main__":    
    
    image_size = 32
    train_loader = load_MNIST("train",image_size)
    test_loader = load_MNIST("test",image_size, random_transform_enabled=False) # the test is without augmentations
    
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    #model = Net(image_size,[512 ,256,512,image_size**2],[]).to(device)
    model = Net(image_size,[512 ,256 ,512,image_size**2],[3,5,1],loss=CrossCorrelationloss).to(device)    
    model.train_model(device,train_loader,test_loader,epoch_num=10)
    
    
    model = model.to('cpu')
    torch.save(model,"model_third_translation_xcorr_loss_n.pt")