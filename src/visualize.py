# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 23:07:02 2023

@author: User
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import datasets,transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.fft import fftshift,fft2

from main_net import Net,dataSet,load_MNIST

model = torch.load('model_quarter_translation_cross_correlation.pt')
model = model.to('cpu')
model.eval()


test_loader = load_MNIST('test',model.image_size, random_transform_enabled=False)
test_fft = test_loader.dataset.data
test_target = test_loader.dataset.targets

model_prediction = model(test_fft)

image_ind = 1

plt.imshow(test_target[image_ind])
# plt.show()
plt.savefig("test_out.png")

plt.imshow(model_prediction[image_ind].detach())
plt.savefig("prediction_out.png")