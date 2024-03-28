import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



class CRNNnet(nn.Module):
    def __init__(self, *, inputLength = 30, kernelSize = 3, kindsOutput = 30):
        super().__init__()
        filterNum1 = 64
        filterNum2 = 32 
        self.layer1 = nn.Sequential(
            nn.Conv1d(100, filterNum1, kernelSize), # inputLength - kernelSize + 1 = 80 - 3 + 1 = 78
            nn.BatchNorm1d(filterNum1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernelSize, stride = 1) # 78 - 3 + 1 = 76
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(filterNum1, filterNum2, kernelSize), # 76 - 3 + 1 = 74
            nn.BatchNorm1d(filterNum2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernelSize, stride = 1) # 74 - 3 + 1 = 72
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(22, kindsOutput)

        self.rnn = nn.GRU(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p, mean=0.0, std=0.001)   # 对参数进行初值化

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
    

model = CRNNnet().to('cuda')
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 0.001)

eta00 = np.load('Main_Project_file/data/data/6000_1s_data_eta00.npy')

# 选取第20个角度的第100到200个所有的时序数据；
data = eta00[:,20,100:200]

for epoch in range(1000):
    
    model.train()
    train_loss = 0
    
    for i in range(4000):
        X = data[0+i:30+i]


        Y = data[30+i:60+i]
        X = np.expand_dims(X, axis=0)
        Y = np.expand_dims(Y, axis=0)


        X_tensor = torch.Tensor(X).to('cuda').permute(0,2,1)
        Y_tensor = torch.Tensor(Y).to('cuda')
        output = model(X_tensor)

        loss = loss_fn(output, Y_tensor)
        train_loss += loss.item() 

        
        optimizer.zero_grad()

        
        loss.backward()


        optimizer.step()

    train_loss =  train_loss / 4000

    print(f'Epoch [{epoch+1}/3000], train Loss: {train_loss}')


    if epoch % 10 == 0:
        model.eval() 
        test_loss = 0
        for i in range(1500):
            X = data[4060+i:4090+i]


            Y = data[4090+i:4120+i]
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)


            X_tensor = torch.Tensor(X).to('cuda').permute(0,2,1)
            Y_tensor = torch.Tensor(Y).to('cuda')
            output = model(X_tensor)
            
            loss = loss_fn(output, Y_tensor)

            test_loss += loss.item()
        test_loss = test_loss / 1500

        print('-------------------------------------------')
        print(f'Epoch [{epoch+1}/3000], test Loss: {test_loss}')
        print('-------------------------------------------')


    if epoch == 200:
        torch.save(model.state_dict(), 'model_parameters200.pth')

    if epoch == 400:
        torch.save(model.state_dict(), 'model_parameters400.pth')

    if epoch == 600:
        torch.save(model.state_dict(), 'model_parameters600.pth')

    if epoch == 800:
        torch.save(model.state_dict(), 'model_parameters800.pth')

torch.save(model.state_dict(), 'model_parameters1000.pth')