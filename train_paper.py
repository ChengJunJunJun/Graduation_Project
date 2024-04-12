import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.crnn import CRNNnet as CRNNnet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMModel(nn.Module):
    import torch.nn.functional as F

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(32, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # LSTM层
        out, _ = self.lstm1(x, (h0, c0))
        out = F.relu(out)
        # 全连接层
        residual = out
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc1(out)
        out = F.relu(out)
        # 残差连接
        out += residual
        # LSTM层
        out, _ = self.lstm2(out, (h0, c0))
        out = F.relu(out)
        # 全连接层
        out = self.fc2(out)
        out = F.relu(out)
        return out

# 定义模型的输入维度、隐藏层维度、隐藏层数、输出维度
input_size = 1
hidden_size = 32
num_layers = 1
output_size = 1

# 创建模型实例
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 打印模型结构
print(model)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    from torchinfo import summary
    summary(model, input_size=[100, 6, 1])

# 均方根误差
def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

# 相关系数
def correlation_coefficient(predictions, targets):
    vx = predictions - torch.mean(predictions)
    vy = targets - torch.mean(targets)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return corr

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 0.001)

data = np.load('data/chengjun_new.npy')
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()


for epoch in range(200):
    
    model.train()
    train_loss = 0
    train_loss_mse = 0
    train_loss_mae = 0
    train_loss_rmse = 0
    train_corr_coef = 0

    for i in range(4000):
        X = data[0+i:6+i]


        Y = data[6+i:12+i]
        X = np.expand_dims(X, axis=0)
        Y = np.expand_dims(Y, axis=0)


        X_tensor = torch.Tensor(X).to('cuda').permute(2,1,0)
        Y_tensor = torch.Tensor(Y).to('cuda').permute(2,1,0)
        output = model(X_tensor)

        loss = loss_fn(output, Y_tensor)
        loss_mse = criterion_mse(output, Y_tensor)
        loss_mae = criterion_mae(output, Y_tensor)
        loss_rmse = rmse(output, Y_tensor)
        corr_coef = correlation_coefficient(output, Y_tensor)
        
        train_loss += loss.item() 
        train_loss_mse += loss_mse.item()
        train_loss_mae += loss_mae.item()
        train_loss_rmse += loss_rmse.item()
        train_corr_coef += corr_coef.item()
        
        optimizer.zero_grad()

        
        loss.backward()


        optimizer.step()
    print('MSE Loss:', train_loss_mse/4000)
    print('MAE Loss:', train_loss_mae/4000)
    print('RMSE Loss:', train_loss_rmse/4000)
    print('Correlation Coefficient:', train_corr_coef/4000)
    print(f'Epoch [{epoch+1}/3000], train Loss: {train_loss/4000}')


    if (epoch + 1) % 10 == 0:
        model.eval() 
        test_loss = 0
        test_loss_mse = 0
        test_loss_mae = 0
        test_loss_rmse = 0
        test_corr_coef = 0
        for i in range(1500):
            X = data[4060+i:4066+i]


            Y = data[4066+i:4012+i]
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)


            X_tensor = torch.Tensor(X).to('cuda').permute(2,1,0)
            Y_tensor = torch.Tensor(Y).to('cuda').permute(2,1,0)
            output = model(X_tensor)
            
            loss = loss_fn(output, Y_tensor)
            loss_mse = criterion_mse(output, Y_tensor)
            loss_mae = criterion_mae(output, Y_tensor)
            loss_rmse = rmse(output, Y_tensor)
            corr_coef = correlation_coefficient(output, Y_tensor)
            test_loss += loss.item()
            test_loss_mse += loss_mse.item()
            test_loss_mae += loss_mae.item()
            test_loss_rmse += loss_rmse.item()
            test_corr_coef += corr_coef.item()
        print('-------------------------------------------')
        print('test MSE Loss:', test_loss_mse/1500)
        print('test MAE Loss:', test_loss_mae/1500)
        print('test RMSE Loss:', test_loss_rmse/1500)
        print('test Correlation Coefficient:', test_corr_coef/1500)
        print(f'Epoch [{epoch+1}/3000], test Loss: {test_loss/1500}')
        print('-------------------------------------------')


    

    
