#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:39:17 2018
With FC layers
@author: Anshul Thakur
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.variable as v

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        ## block 1
        self.conv1 = nn.Conv2d(1, 16, (5,5),padding=2)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, (5,1),stride=(5,1))
       
                ## block 2
        self.conv3 = nn.Conv2d(16, 16, (5,5),padding=2)
        self.conv3_bn = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, (2,1),stride=(2,1))
        
        
                        ## block 3
        self.conv5 = nn.Conv2d(16, 16, (5,5),padding=2)
        self.conv5_bn = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 16, (2,1),stride=(2,1))
        
        
                        ## block 4
        self.conv7 = nn.Conv2d(16, 16, (5,5),padding=2)
        self.conv7_bn = nn.BatchNorm2d(16)
        self.conv8 = nn.Conv2d(16, 16, (2,1),stride=(2,1))
        
        ## reshaping
        
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 2)
    

    def forward(self, x):
        # adding activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_bn(x))
        x = F.relu(self.conv2(x))
        ###########
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3_bn(x))
        x = F.relu(self.conv4(x))
          ###########
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv5_bn(x))
        x = F.relu(self.conv6(x))
          ###########
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv7_bn(x))
        x = F.relu(self.conv8(x))
        
          ############
          # rehsaping and averaging
        x = x.view(-1,16,1000)
        x =F.max_pool1d(x, 1000)
        x = x.view(-1,16)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        #print(x)
        ###############
    
        return x


net = Net()



