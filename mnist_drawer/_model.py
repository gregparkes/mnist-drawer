import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        c1 = 32
        c2 = 64
        input_dim = np.asarray(input_dim)
        d1 = c2 * np.prod((input_dim - 4) // 2)
        d2 = 128

        self.net = nn.Sequential(
            nn.Conv2d(1, c1, 3, 1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(d2, 10)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def name(self):
        return "LeNet"
