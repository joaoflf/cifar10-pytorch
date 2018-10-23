import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as T
import torch.optim as optim
import numpy as np

from models.model import CifarCnn
from data_loaders.cifar10 import Cifar10DataLoader

dtype = torch.cuda.FloatTensor

model = CifarCnn().type(dtype)
loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.Adam(model.parameters(), lr= 1e-3)
data_loader = Cifar10DataLoader()

def train(model, loss_fn, optim, num_epochs=1, print_every=100):
    step = 0
    for epoch in range(num_epochs):
        model.train()
        for t, (x, y) in enumerate(data_loader.train):
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
check_accuracy(model, data_loader.train)
print('Validation Accuracy:')
check_accuracy(model, data_loader.val)
print('Test Accuracy:')
check_accuracy(model, data_loader.test)
