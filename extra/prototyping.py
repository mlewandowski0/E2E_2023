
import torch
from torch import nn
import numpy as np
from torchsummary import summary
from torch import optim
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from vit_pytorch import ViT
from math import sqrt
import torch
from torch.utils.tensorboard import SummaryWriter



from torchviz import make_dot

class smallModelForEnsemble(nn.Module):
    def __init__(self):
        super(smallModelForEnsemble, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 1, bias=False)       # output becomes 26x26
        self.conv1_bn = nn.BatchNorm2d(32)
        self.activation1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 32, 2, bias=False)      # output becomes 24x24
        self.conv2_bn = nn.BatchNorm2d(32)
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 48, 2, bias=False)      # output becomes 24x24
        self.conv3_bn = nn.BatchNorm2d(48)
        self.activation3 = nn.ReLU()

        self.conv4 = nn.Conv2d(48, 48, 3, bias=False)      # output becomes 22x22
        self.conv4_bn = nn.BatchNorm2d(48)
        self.activation4 = nn.ReLU()

        self.conv5 = nn.Conv2d(48, 64, 5, bias=False)      # output becomes 22x22
        self.conv5_bn = nn.BatchNorm2d(64)
        self.activation5 = nn.ReLU()

        self.conv6 = nn.Conv2d(64, 64, 5, bias=False)      # output becomes 22x22
        self.conv6_bn = nn.BatchNorm2d(64)
        self.activation6 = nn.ReLU()

        self.conv7 = nn.Conv2d(64, 64, 5, bias=False)      # output becomes 22x22
        self.conv7_bn = nn.BatchNorm2d(64)
        self.activation7 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(2)
    
    def forward(self, x):
        conv1 = self.activation1(self.conv1_bn(self.conv1(x)))
        conv2 = self.activation2(self.conv2_bn(self.conv2(conv1)))
        conv3 = self.activation3(self.conv3_bn(self.conv3(conv2)))
        conv4 = self.activation4(self.conv4_bn(self.conv4(conv3)))
        conv5 = self.activation5(self.conv5_bn(self.conv5(conv4)))
        conv6 = self.activation6(self.conv6_bn(self.conv6(conv5)))
        conv7 = self.activation7(self.conv7_bn(self.conv7(conv6)))
        conv7 = self.max_pool2(conv7)

        flattened = torch.flatten(conv7.permute(0, 2, 3, 1), 1)
        return flattened
        

class EnsembleModel(nn.Module):
    def __init__(self, model, n, inp):
        super(EnsembleModel, self).__init__()
        self.models = [model() for _ in range(n)]          
        self.logits = nn.Linear(n * self.models[0](inp).shape[-1], 1)
        self.ret_func = nn.Sigmoid() 

    def forward(self, x):
        outs = []
        for model in self.models:
            outs.append(model(x))

        for o in outs:
            print(o.shape)

        layers = torch.cat(outs, dim=-1)
        return self.ret_func(self.logits(layers))
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device Used : {device}")
        
model_ft = EnsembleModel(smallModelForEnsemble, 5, torch.Tensor(1,2,32,32))
model_ft = model_ft.to(device)

print(model_ft(torch.zeros((1,2, 32, 32)).to(device)))