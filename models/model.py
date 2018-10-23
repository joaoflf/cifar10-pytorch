import torch
import torch.nn as nn

# class to flatten layers (eg. Conv to Linear)
class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1) 

# module that composes the overal model
class ConvModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Dropout2d(p=0.3),
                nn.MaxPool2d(2)
             )
    def forward(self, x):
        return self.model(x)
#main model
class CifarCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                ConvModule(3, 32),
                ConvModule(32, 64),
                ConvModule(64, 128),
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
