import argparse
import os
import time
import torch
import matplotlib.pyplot as plt 
import numpy as np
import json

from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image

def validation():
    print("validating parameters")
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")
    if(not os.path.isdir(args.data_directory)):
        raise Exception('directory does not exist!')
    if args.arch not in ('vgg','densenet',None):
        raise Exception('Please choose one of: vgg or densenet')        
    data_dir = os.listdir(args.data_directory)
        

def get_data():
    print("retreiving data")
    train_dir = args.data_directory + '/train'
    test_dir = args.data_directory + '/test'
    valid_dir = args.data_directory + '/valid'
    print("processing data into iterators")
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    loaders = {'train':trainloader,'valid':validloader,'test':testloader,'labels':cat_to_name}
    return loaders

        
def build_model():
    if (args.arch == 'vgg'):
        model = models.vgg16(pretrained=True)
        input_node = 25088
    else:
        model = models.densenet121(pretrained=True)
        input_node = 1024
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    hidden_units = int(args.hidden_units)
    model.classifier = nn.Sequential(nn.Linear(input_node, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))
    return model

def test_accuracy():    
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
                    
            test_loss += batch_loss.item()
                    
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    print(f"Test accuracy: {accuracy/len(testloader):.3f}")
    
def train(model, data):
    print("training model")
    trainloader=data['train']
    validloader=data['valid']
    testloader=data['test']
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    device = torch.device("cuda" if args.gpu else "cpu")
    model.to(device)
    epochs = int(args.epochs)
    lr = float(args.learning_rate)
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    steps = 0
    running_loss = 0
    print_every = 20
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
        # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    print("DONE TRAINING!")
    return(model)
    
def save_model(model):
    print("saving model")
    checkpoint = {'model': model.cpu(),
                  'features': model.features,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, args.save_dir)
    print('Congratulation! Model has been saved to ', args.save_dir)

def create_model():
    validation()
    data = get_data()
    model = build_model()
    model = train(model, data)
    save_model(model)
    return None

def parse():
    parser = argparse.ArgumentParser(description='You can input several arguments:')
    parser.add_argument('data_directory', help='data directory (required)')
    parser.add_argument('--save_dir', default='checkpoint.pth', help='directory to save a neural network.')
    parser.add_argument('--arch', default='vgg', help='models to use OPTIONS[vgg, densenet]')
    parser.add_argument('--learning_rate', default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', default=4096, help='number of hidden units')
    parser.add_argument('--epochs', default=3, help='epochs, default=3')
    parser.add_argument('--gpu', default=True, action='store_true', help='gpu')
    args = parser.parse_args()
    return args

def main():
    print("Creating a deep learning model")
    global args
    args = parse()
    create_model()
    print("MODEL DONE!")

main()