import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import tensorly as tl
from tensorly.tenalg import inner
from tensorly.random import check_random_state
tl.set_backend('pytorch')
import numpy as np

class TRL(nn.Module):
    def __init__(self, input_size, ranks, output_size, verbose=1, **kwargs):
        super(TRL, self).__init__(**kwargs)
        self.ranks = list(ranks)
        self.verbose = verbose

        if isinstance(input_size, int):
            self.input_size = [input_size]
        else:
            self.input_size = list(input_size)
            
        if isinstance(output_size, int):
            self.output_size = [output_size]
        else:
            self.output_size = list(output_size)
            
        self.n_outputs = int(np.prod(output_size[1:]))
        
        # Core of the regression tensor weights
        self.core = nn.Parameter(tl.zeros(self.ranks), requires_grad=True)
        self.bias = nn.Parameter(tl.zeros(1), requires_grad=True)
        weight_size = list(self.input_size[1:]) + list(self.output_size[1:])
        
        # Add and register the factors
        self.factors = []
        for index, (in_size, rank) in enumerate(zip(weight_size, ranks)):
            self.factors.append(nn.Parameter(tl.zeros((in_size, rank)), requires_grad=True))
            self.register_parameter('factor_{}'.format(index), self.factors[index])
        
        # FIX THIS
        self.core.data.uniform_(-0.1, 0.1)
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        regression_weights = tl.tucker_to_tensor((self.core, self.factors))
        return inner(x, regression_weights, n_modes=tl.ndim(x)-1) + self.bias
    
    def penalty(self, order=2): #has problem
        penalty = tl.norm(self.core, order)
        for f in self.factors:
            penalty = penalty + tl.norm(f, order)
        return penalty


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        trainbatch = 32
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.3)
        self.conv3 = nn.Conv2d(64, 64, 3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.trl = TRL(ranks=(30, 10, 10, 10), input_size=(trainbatch, 64, 128, 128), output_size=(trainbatch,130))

    def forward(self, x):
        x = (F.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)
        x = (F.relu(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = (F.relu(self.bn3(self.conv3(x))))
        x = self.trl(x)
        return F.log_softmax(x, dim=1)