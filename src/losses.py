import torch
import numpy as np

import torch.nn as nn
from torch.fft import fftshift,fft2,ifft2
import torch.nn.functional as F

def MSEloss(output: torch.Tensor, target: torch.Tensor):
    return nn.MSELoss()(output, target)

def Metricloss(output: torch.Tensor, target: torch.Tensor , p: int=2):
    return torch.mean(Metricbatch(output,target,p))

def MSEbatchloss(output: torch.Tensor, target: torch.Tensor):
    return Metricbatch(output,target,2)

def Metricbatch(output: torch.Tensor, target: torch.Tensor, p: int=2) -> torch.Tensor:
    """

    metric of batch of images - compute sum over all dimensions except batch dimension.

    Args:
        output (torch.Tensor): output tensor
        target (torch.Tensor): target tensor
        p (int, optional): degree. Defaults to 2.

    Returns:
        torch.Tensor: mse loss
    """
    input_dims = tuple([i + 1 for i in range(target.dim() - 1)]) # to comply with pytorch syntax. all dimensions except (first) batch dimension
    m = torch.sum((torch.abs(output - target))**p, dim=input_dims) / (torch.numel(output) / output.size(0))
    return m
    
def RotationInvariantMSEloss(output: torch.Tensor, target: torch.Tensor):
    """wrapper"""
    return RotationInvariantMetric(output,target,2)


def RotationInvariantMetric(output: torch.Tensor, target: torch.Tensor, p: int=2) -> torch.Tensor:
    """rotation invariant metric (mse) loss"""
    return torch.mean(torch.min(Metricbatch(output,target,p), Metricbatch(torch.rot90(output,2,(-2,-1)), target,p)))


def RotationTranslationInvariantMSEloss(output: torch.Tensor, target: torch.Tensor, translation_limit:int = None):
    """wrapper"""
    return RotationTranslationInvariantMetric(output,target,translation_limit,2)


def RotationTranslationInvariantMetric(output: torch.Tensor, target: torch.Tensor,translation_limit: int = None, p = 2) -> torch.Tensor:
    """

    rotation and translation invariant metric (mse) loss

    Args:
        output (torch.Tensor): output tensor
        target (torch.Tensor): target tensor
        translation_limit (int, optional): pixel translation limit. Defaults to None.
        p (int, optional): degree. Defaults to 2.

    Returns:
        torch.Tensor: loss
    """
    image_size = output.size(1)
   
    if(translation_limit == None):
        translation_limit = image_size
    # 2 x translation_limit x translation_limit x batch_size
    loss_rot_trans = torch.zeros(2, translation_limit, translation_limit, output.size(0)) 
    
    # loop over translation limit
    for i in range(translation_limit):
        for j in range(translation_limit):
            # roll by i,j over 2nd and 3rd dimensions
            shifted_output = torch.roll(output,(i,j),(1,2))
            
            # compute loss for original and rotated image
            loss_rot_trans[0,i,j,:] = Metricbatch(shifted_output,target,p)
            shifted_rot_output = torch.rot90(shifted_output,2,(-2,-1))
            loss_rot_trans[1,i,j,:] = Metricbatch(shifted_rot_output,target,p)
    
    # permute to get batch_size x 2 x translation_limit x translation_limit
    loss_rot_trans = loss_rot_trans.permute(3,0,1,2)
    loss_rot_trans = loss_rot_trans.view(loss_rot_trans.size(0),-1)
    
    loss = torch.mean(torch.min(loss_rot_trans,dim=1)[0])
    return loss

def AlignPrediction(output: torch.Tensor, target: torch.Tensor, rot180: boo = False, circ_shift: bool = False):
    # Aligns prediction image with target image (to be used later in performance metric calculation)
    
    if((not rot180) and (not circ_shift)): # If no rotation or translation needed
        return output
    
    if(rot180 and (not circ_shift)):
        mse = MSEbatchloss(output,target)
        mse_rot = MSEbatchloss(torch.rot90(output,2,(-2,-1)), target)
        # 0 - no rotation, 1 - 180 degree rotation
        pred_rot_val = torch.argmin(torch.stack((mse,mse_rot)),dim=0)
        pred_rot = torch.clone(output)
        # rotate images by 180 degrees if needed
        for i in range(output.size(0)):
            pred_rot[i] = torch.rot90(output[i],2*pred_rot_val[i],(-2,-1))
            
        return pred_rot
    
    if(circ_shift):
        
        if(output.size(0) > 20): #perform function in batches to prevent memory issues
            ind_start = lambda i : i*20
            ind_end = lambda i : min((i+1)*20,output.size(0))
            batch_alignment = [AlignPrediction(output[ind_start(i):ind_end(i)],target[ind_start(i):ind_end(i)],rot180 ,circ_shift) for i in range(int(np.ceil(output.size(0)/20)))]
            return torch.cat(batch_alignment)
        
        translation_limit = output.size(1)
        loss_shift = torch.zeros(translation_limit,translation_limit,output.size(0))
        pred_shift = torch.clone(output)
        
        if(rot180):
            loss_rot_shift = torch.zeros(translation_limit,translation_limit,output.size(0))
        
        for i in range(translation_limit):
            for j in range(translation_limit):
                shifted_output = torch.roll(output,(i,j),(1,2))
                loss_shift[i,j,:] = MSEbatchloss(shifted_output,target)
                
                if(rot180):
                    shifted_rot_output = torch.rot90(shifted_output,2,(-2,-1))
                    loss_rot_shift[i,j,:] = MSEbatchloss(shifted_rot_output,target)
        
       
        pred_shift_val = np.unravel_index(torch.argmin(loss_shift.view(-1,loss_shift.size(2)),dim=0),(loss_shift.size(0),loss_shift.size(1)))
        if(rot180):
            pred_rot_shift_val = np.unravel_index(torch.argmin(loss_rot_shift.view(-1,loss_rot_shift.size(2)),dim=0),(loss_rot_shift.size(0),loss_rot_shift.size(1)))
            
        
        for i in range(output.size(0)):
            if(not rot180):
                roll_val = (pred_shift_val[0][i],pred_shift_val[1][i])
                pred_shift[i] = torch.roll(output[i],roll_val)
            else:
                roll_val = (pred_shift_val[0][i],pred_shift_val[1][i])
                roll_rot_val = (pred_rot_shift_val[0][i],pred_rot_shift_val[1][i])
                if(loss_shift[roll_val[0],roll_val[1],i] < loss_rot_shift[roll_rot_val[0],roll_rot_val[1],i]):
                    pred_shift[i] = torch.roll(output[i],roll_val,dims=(0,1))
                else:
                    pred_shift[i] = torch.rot90(torch.roll(output[i],roll_rot_val,dims=(0,1)),2,(-2,-1))
                    
        return pred_shift
    

def L2NormMetric(output,target):
    return torch.mean(torch.sqrt(MSEbatchloss(output, target)))/np.sqrt((torch.numel(output)/output.size(0)))

def RotationInvariantL2NormMetric(output,target):
    return torch.mean(torch.sqrt(torch.min(MSEbatchloss(output,target),MSEbatchloss(torch.rot90(output,2,(-2,-1)), target))))/np.sqrt((torch.numel(output)/output.size(0)))


def CrossCorrelationloss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """

    cross correlation loss

    Args:
        output (torch.Tensor): output tensor
        target (torch.Tensor): target tensor

    Returns:
        torch.Tensor: x-correlation loss
    """
    fft_output = fft2(output)
    fft_target = fft2(target)
    correlation = torch.abs(ifft2(torch.mul(fft_output,torch.conj(fft_target))))
    max_corr = torch.max(correlation.view(correlation.size(0), -1), dim=1)[0]
    
    fft_output_rot = fft2(torch.rot90(output,2,(-2,-1)))
    correlation_rot = torch.abs(ifft2(torch.mul(fft_output_rot,torch.conj(fft_target))))
    max_corr_rot = torch.max(correlation_rot.view(correlation_rot.size(0),-1),dim=1)[0]
    
    max_corr_total = torch.max(max_corr, max_corr_rot)
    
    output_target_norm_prod = torch.mul(torch.norm(output,p=2,dim=(1,2)),torch.norm(target,p=2,dim=(1,2)))
    max_corr_total = torch.div(max_corr_total, output_target_norm_prod)
    
    final_corr_value = torch.sum(max_corr_total)
    
    return (1 / final_corr_value) 


def MagnitudeMSEloss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """

    magnitude mse loss

    Args:
        output (torch.Tensor): output tensor
        target (torch.Tensor): target tensor

    Returns:
        torch.Tensor: desired loss
    """
    mag_output = torch.abs(fft2(output))
    mag_target = torch.abs(fft2(target))
  
    return nn.MSELoss()(mag_output,mag_target)/(output.size(1)*output.size(2)) #Normalize fft bins


def CrossCorrelation_MagnitudeMSE(output: torch.Tensor, target: torch.Tensor, lambda_val: float = 0.1) -> torch.Tensor:
    """

    cross correlation + magnitude mse weighted loss

    Args:
        output (torch.Tensor): output tensor
        target (torch.Tensor): target tensor
        lambda_val (float, optional): magnitude loss weight. Defaults to 0.1.

    Returns:
        torch.Tensor: desired loss
    """
    return CrossCorrelationloss(output,target) + lambda_val * MagnitudeMSEloss(output,target)
    