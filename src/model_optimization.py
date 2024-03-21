from os.path import isfile,join
import torch
import numpy as np
import copy
import torch.nn as nn
from torchvision import datasets,transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.fft import fftshift,fft2

from torch.optim import lr_scheduler 

from losses import *
from main_net import saveModel

def train_save_model(model_name,data_name,model,device,train_loader,test_loader,epoch_num=10,learning_rate = 5e-5):
    """
        Trains and Saves model to file

        Args:
            model_name (string) : name of model
            data_name (string) : name of dataset ('MNIST' or 'Fashion-MNIST')
            model (nn.Module) : model to be trained
            device (string) : device to load model to
            train_loader (DataLoader) : data loader of training data
            test_loader (DataLoader) : data loader of testing data
            epoch_num (int) : number of epochs to train
            learning_rate (float) : learning rate of optimizer 
            
           
    """
    if(not('.pt' in model_name)):
        model_name +='.pt'
        
    if(isfile(join('Models',data_name,model_name))): #Checks wether a model with same name is already saved, if this is the case the function returns without training the new model
        print(f'{model_name} with {data_name} data is already trained and saved, delete the file {join("Models",data_name,model_name)} to retrain the model \n')
        return
    train_model(model,device,train_loader,test_loader,epoch_num,learning_rate)
    saveModel(model,model_name,data_name) 

def train_model(model,device,train_loader,test_loader,epoch_num=10,learning_rate = 5e-5):
    """
        Trains Model

        Args:
            model (nn.Module) : model to be trained
            device (string) : device to load model to
            train_loader (DataLoader) : data loader of training data
            test_loader (DataLoader) : data loader of testing data
            epoch_num (int) : number of epochs to train
            learning_rate (float) : learning rate of optimizer 
            
           
    """
    optimizer = optim.Adam(model.parameters(), learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [25,50,70,80],gamma=0.5)
    model.to(device)
    model.train()
    for i in range(1,epoch_num+1):
        train_model_epoch(model,device,train_loader,optimizer,i)
        test_model(model,device,test_loader)
        scheduler.step()
    model.eval()
    
    
def train_model_epoch(model, device, train_loader, optimizer, epoch):
    """
        Trains a single epoch of a Model

        Args:
            model (nn.Module) : model to be trained
            device (string) : device to load model to
            train_loader (DataLoader) : data loader of training data
            optimizer (torch.optim) : optimizer that is used for training
            epoch (int) : current epoch number
            
           
    """
    model.train()    

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
               
        loss = model.compute_loss(data,target)
        #loss.requires_grad = True
            
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader)),end='\r')


def test_model(model,device,test_loader):
    """
        Tests model with its loss function 

        Args:
            model (nn.Module) : model to be trained
            device (string) : device to load model to
            test_loader (DataLoader) : data loader of test data
      
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            test_loss += model.compute_loss(data,target).item()  * data.size(0)
            


    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss, correct, len(test_loader.dataset)))
    
    
def model_optimization(model,device,train_loader,test_loader,epoch_num=10,learning_rate = 5e-5,lambda_vals = [],lambda_field='rep_loss_lambda',metricFun = MSEloss):
    """
        Trains Model with different lambda values (weight of additional loss function) and returns the model with best results

        Args:
            model (nn.Module) : model to be trained
            device (string) : device to load model to
            train_loader (DataLoader) : data loader of training data
            test_loader (DataLoader) : data loader of testing data
            epoch_num (int) : number of epochs to train
            learning_rate (float) : learning rate of optimizer 
            lambda_vals (list[float]) : lambda values to be used for training and testing
            lambda_field (string) : the field name of the model that corresponds to the value of lambda  
            metricFun : function that measures results of the model, the returned model is the model who has the minimal result of metricFun
            
           
    """
    optimal_metric = torch.inf
    optimal_lambda_value = -1
    validation_samples_data = test_loader.dataset.data[:1000].to(device)
    validation_samples_target = test_loader.dataset.targets[:1000].to(device)
    model.to(device)
    lambda_metrics = []
    for lambda_val in lambda_vals:
        print('\nTraining with lambda value {:.4f}'.format(lambda_val))
        tmp_model = copy.deepcopy(model)
        exec('tmp_model.' + lambda_field + ' = ' + str(lambda_val)) #update lambda paramter of model
        
        train_model(tmp_model,device,train_loader,test_loader,epoch_num,learning_rate)
        
        tmp_metric = metricFun(tmp_model.predict(validation_samples_data),validation_samples_target)
        lambda_metrics.append(tmp_metric.item())
        if(tmp_metric < optimal_metric):
            optimal_metric = tmp_metric
            optimal_model = copy.deepcopy(tmp_model)
            optimal_lambda_value = lambda_val
            
            
        print('\nNew Metric Value {:4E},Optimal Metric Value {:.4E}, optimal lambda value {:.4E} \n'.format(tmp_metric,optimal_metric,optimal_lambda_value))
        
   
    return optimal_model,lambda_metrics