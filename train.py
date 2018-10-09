#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:16:02 2018

@author: Anshul Thakur
"""
from sklearn.model_selection import train_test_split
from all_conv import Net
import torch.utils.data as utils
import numpy as np
from keras.utils import to_categorical
import torch
import torch.utils.data as utils
from torchvision import transforms
from all_conv_2 import Net
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import time




feature_ff = np.load('ff_20_10_feature.npy')
label_ff = np.load('ff_20_10_label.npy')
feature_wab = np.load('wblr_20_10_feature.npy')
label_wab= np.load('wblr_20_10_label.npy')


feature=np.concatenate((feature_ff, feature_wab), axis=0)
feature=feature.reshape(-1,1,40,1000)
print(feature.shape)

label=np.concatenate((label_ff, label_wab), axis=0)



label=label.astype(int)

print(label.shape)
label=to_categorical(label)

## divide into train and val


x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.35, shuffle=True)

y_train=torch.FloatTensor(torch.from_numpy(y_train).float())
y_test=torch.FloatTensor(torch.from_numpy(y_test).float())


 # transform to torch tensors
train = torch.stack([torch.Tensor(i) for i in x_train])
#train_labels = torch.stack([torch.Tensor(i) for i in y_train])
test = torch.stack([torch.Tensor(i) for i in x_test])
#test_labels = torch.stack([torch.Tensor(i) for i in y_test])

train_labels=torch.LongTensor(y_train.long())
test_labels=torch.LongTensor(y_test.long())


## creater dataloaders
train_dataset = utils.TensorDataset(train,train_labels) # create your datset
test_dataset=utils.TensorDataset(test,test_labels)
train_dataloader = utils.DataLoader(train_dataset,shuffle=True,batch_size=32,num_workers=1) # create your dataloader
test_dataloader = utils.DataLoader(test_dataset,shuffle=True,batch_size=32,num_workers=1)


## create final dataloader
dataset_sizes = {'train':len(train_dataloader.dataset),'valid':len(test_dataloader.dataset)}
dataloaders = {'train':train_dataloader,'valid':test_dataloader}

################

net=Net()
net=net.cuda()
###########
# parameters

#learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)




print('before training')

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                #print(inputs.size())
                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                x, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, torch.max(labels, 1)[1])
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                #print('preds')
                #print(preds)
                #print('labels')
                #print(labels.data)
                #print('loss_data')
                #print(loss.data)   
                running_loss += loss.data
                running_corrects += torch.sum(preds == torch.max(labels, 1)[1])
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.data.cpu().numpy() / dataset_sizes[phase]
            print('acc_probs')
            print(running_corrects.data.cpu().numpy())
            print(dataset_sizes[phase])             
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
      #  print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = train_model(net, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)



torch.save(model_ft, 'all_conv_example_2.pt')


















