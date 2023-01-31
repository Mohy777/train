# Imports here

#% matplotlib inline
#% config InlineBackend.figure_format='retina'

import argparse

import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

from workspace_utils import active_session

from PIL import Image


import os

import numpy as np

def get_input_args():
    
    parser=argparse.ArgumentParser(description='train.py')
    
    parser.add_argument('--data_dir', type=str, default='flowers' , help='path to folder of images')
    
    parser.add_argument('--save_dir', type=str, default='save_directory', help='path to the folder to save checkpoints')
    
    parser.add_argument('--arch', choices=['vgg16', 'densenet'], type=str, default='vgg16', help='model architecture')
    
    parser.add_argument('--gpu', action='store_true', default=False, help='Use gpu for training, defaults to False')
    
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning_rate')
    
      
    parser.add_argument('--epochs', type=int, default= 1, help='epochs')
    
    parser.add_argument('--path/to/image', type=str, default='flowers/test/12', help='input image to be classified')
    
    return parser.parse_args()

args=get_input_args()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])




train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# Build and train your network
if args.arch == 'vgg16':
    model=models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad=False
    
    classifier=nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088,4096)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.3)),
        ('fc2', nn.Linear(4096,1000)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.3)),
        ('fc3', nn.Linear(1000,102)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    
elif args.arch == 'densenet':
    model=model.densenet161(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad=False
    
    classifier=nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2208,1000)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.3)),
        ('fc2', nn.Linear(1000,500)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.3)),
        ('fc3', nn.Linear(500,102)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
        
model.classifier=classifier
criterion=nn.NLLLoss()

lr=args.learning_rate
optimizer=optim.Adam(model.classifier.parameters(), lr)

model.to(device);

#train the model
epochs=args.epochs
steps = 0
running_loss = 0
print_every = 5

with active_session():
    for epoch in range (epochs):
        for images, labels in trainloader:
            steps+=1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            output=model.forward(images)
            loss=criterion(output, labels)
            
            loss.backward()
            optimizer.step()
        
            running_loss+=loss.item() 
            
            if steps%print_every ==0:
                test_loss=0
                accuracy=0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps=model.forward(inputs)
                        batch_loss=criterion(logps, labels)
                    
                        test_loss+=batch_loss.item()
                    
                        ps=torch.exp(logps)
                        top_p, top_class=ps.topk(1, dim=1)
                        equals=top_class == labels.view(*top_class.shape)
                        accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                
                running_loss = 0
                model.train()
                    
# Do validation on the test set
valid_loss=0
accuracy=0
model.eval()
with torch.no_grad():
    for inputs, labels in validloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps=model.forward(inputs)
        batch_loss=criterion(logps, labels)
                    
        valid_loss+=batch_loss.item()
                    
        ps=torch.exp(logps)
        top_p, top_class=ps.topk(1, dim=1)
        equals=top_class == labels.view(*top_class.shape)
        
        accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
            
    print(f"Validation loss: {valid_loss/len(validloader):.3f}.. "
          f"Validation accuracy: {accuracy/len(validloader):.3f}")  
 

# Save the checkpoint 
model.class_to_idx=train_data.class_to_idx

if args.arch == 'vgg16':
    checkpoint={'input_size':25088,
                'output_size':102,
                'epochs':1,
                'mapping_class_to_idx':model.class_to_idx,
                'optimizer_state':optimizer.state_dict(),
                'state_dict':model.state_dict(),
                'classifier':model.classifier}
    
elif args.arch == 'densenet':
    checkpoint={'input_size':2208,
                'output_size':102,
                'epochs':1,
                'mapping_class_to_idx':model.class_to_idx,
                'optimizer_state':optimizer.state_dict(),
                'state_dict':model.state_dict(),
                'classifier':model.classifier}


checkpoint_path=args.save_dir
torch.save(checkpoint, checkpoint_path)
