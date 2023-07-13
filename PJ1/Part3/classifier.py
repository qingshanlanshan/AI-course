# %% [markdown]
# # CNN Network

# %%
import torch
import torch.nn.functional as F
import os
import numpy as np
from typing import Tuple
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import random
from collections import OrderedDict

print(torch.__version__)
print(torchvision.__version__)

curPath=os.path.abspath('')
dataPath=curPath+"/../data"
BATCH_SIZE=64
EPOCHS=1
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
DEVICE = torch.device("mps") 
print(DEVICE)
print(dataPath)


# %%
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    
])

train_set = datasets.MNIST(root=dataPath, train=True, download=True, transform=train_transform)
test_set = datasets.MNIST(root=dataPath, train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride= 2, padding=3, bias= False)
# model.fc = torch.nn.Sequential(OrderedDict([('fc1', torch.nn.Linear(model.fc.in_features, 10)),
                                            # ('output', torch.nn.Softmax(dim=1))
                                            # ]))
# model.fc=torch.nn.Linear(model.fc.in_features,10)
# torch.nn.init.xavier_uniform_(model.fc.weight)

model.fc = torch.nn.Sequential(OrderedDict([('fc1', torch.nn.Linear(model.fc.in_features, 64)),
                                            ('relu1', torch.nn.ReLU()), 
                                            ('dropout',torch.nn.Dropout(0.5)),
                                            ('fc2', torch.nn.Linear(64, 10)),
                                            # ('output', torch.nn.Softmax(dim=1))
                                            ]))
torch.nn.init.xavier_uniform_(model.fc[0].weight)
torch.nn.init.xavier_uniform_(model.fc[3].weight)


criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
# optimizer = torch.optim.Adam([{'params':model.fc.parameters()}], lr=0.00002)
optimizer = torch.optim.SGD([{'params':model.fc.parameters()}], lr=0.00001)
# StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

model=model.to(DEVICE)
print(model)
print(len(train_loader))

# %%
def test(model, device, test_loader):
    model.eval()
    sum_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).to(DEVICE)
            loss = criterion(output, target)  # calculate loss
            sum_loss += loss.item()  # add loss to running total
            pred = output.argmax(dim=1, keepdim=True)  # get predicted class
            correct += pred.eq(target.view_as(pred)).sum().item()  # count correct predictions

    sum_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        sum_loss, correct, len(test_loader.dataset), accuracy))
    return sum_loss, accuracy
def train(model,device,train_loader,epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(data).to(DEVICE)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if batch_idx%(len(train_loader)//4)==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


# %%
# load the saved model
model.load_state_dict(torch.load('97-5-multi-linear-xavier-dropout.pt',map_location=DEVICE))

# %%
for epoch in range(EPOCHS):
    train(model, DEVICE, train_loader, epoch)
    print("Testing...")
    # print("Train set:")
    # test(model, DEVICE, train_loader)
    print("Test set:")
    loss,accuracy=test(model, DEVICE, test_loader)
    if accuracy>=97.50:
        torch.save(model.state_dict(), '97-5-multi-linear-xavier-dropout.pt')
        break

# %%
test(model, DEVICE, train_loader)
test(model, DEVICE, test_loader)

# %%
# save the model
torch.save(model.state_dict(), '97-5-multi-linear-xavier-dropout.pt')


