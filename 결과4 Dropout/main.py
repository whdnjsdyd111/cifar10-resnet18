import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os
from tqdm import tqdm


def createDirectory(dr):
    try:
        if not os.path.exists(dr):
            os.makedirs(dr)
    except OSError:
        print('error create dir')


parser = argparse.ArgumentParser()

parser.add_argument('--model_num', type=int, default=3, help='model number')
parser.add_argument('--total_epoch', type=int, default=50, help='epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batches_train', type=int, default=128, help='batch size train')
parser.add_argument('--batches_eval', type=int, default=100, help='batch size eval')

args = parser.parse_args()

model_num = args.model_num # total number of models
total_epoch = args.total_epoch # total epoch
lr = args.lr # initial learning rate
batch_size_train = args.batches_train
batch_size_eval = args.batches_eval


class Resnet18(nn.Module):
    def __init__(self, pretrained_model, num_classes, dr_rate):
        super(Resnet18, self).__init__()
        self.resnet = pretrained_model
        ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(ftrs, num_classes)

        self.dropout = nn.Dropout(p=dr_rate)


    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)

        return x


for s in range(model_num):
    # fix random seed
    seed_number = s
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define the data transforms
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_eval, shuffle=False, num_workers=2)

    # Define the ResNet-18 model with pre-trained weights
    pretrained_model = timm.create_model('resnet18', pretrained=True)
    model = Resnet18(pretrained_model, 10, 0.2)


    # Move the model to the GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)  
    else:
        model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    def train():
        model.train()
        _loss = 0
        _correct = 0
        _total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            
            loss = criterion(outputs, labels)

            l1_lambda = 0.0001
            l1_norm = torch.tensor(0.).to(device)
            for param in model.parameters():
                l1_norm += torch.norm(param, p=1)

            loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()
            
            _loss += loss.item()
            _total += labels.size(0)
            _correct += (predicted == labels).sum().item()
            
        
        return _loss / _total, _correct / _total * 100
    
    
    def test():
        model.eval()
        _loss = 0
        _correct = 0
        _total = 0
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
                
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)
                
                _loss += loss.item()
                _total += labels.size(0)
                _correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %f %%' % (100 * _correct / _total))
        
        return _loss / _total, _correct / _total * 100

    rc_txt = './epoch-%d lr-%.4f batch-%d model_num%d-%d.txt' % (total_epoch, lr, batch_size_train, model_num, seed_number)

    if os.path.exists(rc_txt) == False:
        with open(rc_txt, 'w') as f:
            f.write('Start!')

    tl_list = []
    ta_list = []
    el_list = []
    ea_list = []

    # Train the model
    for epoch in tqdm(range(total_epoch)):
        t_loss, t_acc = train()
        e_loss, e_acc = test()
        scheduler.step()
        
        tl_list.append(t_loss)
        ta_list.append(t_acc)
        el_list.append(e_loss)
        ea_list.append(e_acc)
        
        with open(rc_txt, 'a') as f:
            f.write('\nt_loss: %7.3f t_acc: %5.3f e_loss: %7.3f e_acc: %5.3f' % (t_loss, t_acc, e_loss, e_acc))
        
        

    print('Finished Training')

    dr = './resnet18_cifar10/epochs-%d/lr-%.4f/batch-%d/' % (total_epoch, lr, batch_size_train)

    # Save the checkpoint of the last model
    createDirectory(dr)
    PATH = dr + 'model_num-%d-%d.pth' % (model_num, seed_number)
    torch.save(model.state_dict(), PATH)
    
    plt.plot(tl_list, label='train loss', marker='.')
    plt.plot(el_list, label='test loss', marker='.')
    plt.legend()
    plt.savefig('epoch-%d lr-%.4f batch-%d model_num%d-%d loss.png' % (total_epoch, lr, batch_size_train, model_num, seed_number))
    plt.clf()

    plt.plot(ta_list, label='train acc', marker='.')
    plt.plot(ea_list, label='test acc', marker='.')
    plt.legend()
    plt.savefig('epoch-%d lr-%.4f batch-%d model_num%d-%d acc.png' % (total_epoch, lr, batch_size_train, model_num, seed_number))
    plt.clf()
    
    print('epoch-%d lr-%.4f batch-%d model_num%d-%d Saved!' % (total_epoch, lr, batch_size_train, model_num, seed_number))
    
    
