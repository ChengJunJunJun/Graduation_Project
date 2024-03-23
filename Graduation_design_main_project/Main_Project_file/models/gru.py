import os
import sys
import time
import torch.nn as nn
import torch
from torchinfo import summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from utils.time import time_calc

class GruNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size ):
        super(GruNet, self).__init__() 

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)  
        out = self.linear(out)
        return out





if __name__ == '__main__':
    X = torch.randn(320,10,1)
    # hidden_prev = torch.randn(1,20,16)
    model = GruNet(input_size=1, hidden_size=256, output_size=1)
    pred = model(X)
    print(model)
    summary(model, input_size=[320, 10, 1])

