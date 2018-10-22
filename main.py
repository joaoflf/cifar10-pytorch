import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.optim as optim

import numpy as np

# Sampler to iterate on dataset, given size and start point
class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))
    
    def __len__(self):
        return self.num_samples

NUM_TRAIN = 49000
NUM_VAL = 1000
dtype = torch.cuda.FloatTensor

# setup train, val and test set
train_set = dset.CIFAR10('./datasets', train=True, download=True, transform=T.ToTensor())
loader_train = DataLoader(train_set, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))
val_set = dset.CIFAR10('./datasets', train=True, download=True, transform=T.ToTensor())
loader_val = DataLoader(val_set, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))
test_set = dset.CIFAR10('./datasets', train=False, download=True, transform=T.ToTensor())
loader_val = DataLoader(test_set, batch_size=64)

# class to flatten layers (eg. Conv to Linear)
class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1) 
# model
class CifarCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout2d(p=0.3),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.Dropout2d(p=0.3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.Dropout2d(p=0.3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                Flatten(),
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 10)
            )
        self.model.apply(self.init_weights)

    def forward(self, x):
        return self.model(x)
    
    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

model = CifarCnn().type(dtype)
loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.Adam(model.parameters(), lr= 1e-3)

def train(model, loss_fn, optim, num_epochs=1, print_every=100):
    step = 0
    for epoch in range(num_epochs):
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype).long())

            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            if (step % print_every == 0) and step>0:
                print('t = %d, loss = %.4f' % (step, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step+=1
def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    for x, y in loader:
        x_var = Variable(x.type(dtype))
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    acc = float(num_correct)/num_samples
    print('Got %d / %d correct (%.2f)%%' % (num_correct, num_samples, 100 * acc))

train(model, loss_fn, optim, 20)
print('Train Accuracy:')
check_accuracy(model, loader_train)
print('Validation Accuracy:')
check_accuracy(model, loader_val)
