import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

decice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RCNNnet(nn.Module):
    def __init__(self, input_size = 100, hidden_size = 256, outputsize = 100, kernelSize = 3):
        super().__init__() 

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        filterNum1 = 128
        filterNum2 = 100
        self.layer1 = nn.Sequential(
            nn.Conv1d(256, filterNum1, kernelSize), 
            nn.BatchNorm1d(filterNum1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernelSize, stride = 1) 
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(filterNum1, filterNum2, kernelSize),
            nn.BatchNorm1d(filterNum2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernelSize, stride = 1) 
        )
        self.dropout = nn.Dropout(0.2)
        

        self.linear = nn.Linear(22, 1)

    def forward(self, x):
        out, _ = self.rnn(x)  

        out = out.permute(0, 2, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.linear(out)
        out = self.dropout(out)
        return out
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RCNNnet().to(device)
    from torchinfo import summary
    summary(model, input_size=[1,30,100])