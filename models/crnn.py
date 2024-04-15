import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



class CRNNnet(nn.Module):
    def __init__(self,  inputLength = 30, kernelSize = 3, kindsOutput = 30):
        super().__init__()
        filterNum1 = 64
        filterNum2 = 32 
        self.layer1 = nn.Sequential(
            nn.Conv1d(100, filterNum1, kernelSize), 
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
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(52, 60)

        self.rnn = nn.GRU(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)  

        self.linear = nn.Linear(64, 100)
    
    def forward(self,x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        x = x.permute(0,2,1)
        out, _ = self.rnn(x)  # [b, seq, h]
        out = self.linear(out)
        return out
    

class CLSTMnet(nn.Module):
    def __init__(self,  inputLength = 30, kernelSize = 3, kindsOutput = 30):
        super().__init__()
        filterNum1 = 64
        filterNum2 = 32 
        self.layer1 = nn.Sequential(
            nn.Conv1d(100, filterNum1, kernelSize), 
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
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(52, 60)

        self.rnn = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)  

        self.linear = nn.Linear(64, 100)
    
    def forward(self,x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        x = x.permute(0,2,1)
        out, _ = self.rnn(x)  # [b, seq, h]
        out = self.linear(out)
        return out

class CNNnet(nn.Module):
    def __init__(self, *, inputLength = 30, kernelSize = 3, kindsOutput = 100):
        super().__init__()
        filterNum1 = 64
        filterNum2 = 32 
        self.layer1 = nn.Sequential(
            nn.Conv1d(100, filterNum1, kernelSize), 
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
        self.fc = nn.Linear(32, kindsOutput)

    def forward(self,x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x
    
class RNNnet(nn.Module):
    def __init__(self, input_size = 100, hidden_size = 256, outputsize = 100):
        super().__init__() 

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)   # 对参数进行初值化

        self.linear = nn.Linear(hidden_size, outputsize)

    def forward(self, x):
        out, _ = self.rnn(x)  
        out = self.linear(out)
        return out
    

class GRUnet(nn.Module):
    def __init__(self, input_size = 100, hidden_size = 256, outputsize = 100):
        super().__init__() 

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)   # 对参数进行初值化

        self.linear = nn.Linear(hidden_size, outputsize)

    def forward(self, x):
        out, _ = self.rnn(x)  
        out = self.linear(out)
        return out


class LSTMnet(nn.Module):
    def __init__(self, input_size = 100, hidden_size = 256, outputsize = 100):
        super().__init__() 

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)   # 对参数进行初值化

        self.linear = nn.Linear(hidden_size, outputsize)

    def forward(self, x):
        out, _ = self.rnn(x)  
        out = self.linear(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLSTMnet().to(device)
    from torchinfo import summary
    summary(model, input_size=[1,100,60])