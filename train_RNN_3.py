import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.crnn import RNNnet , GRUnet, LSTMnet


# 均方根误差
def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))

# 相关系数
def correlation_coefficient(predictions, targets):
    vx = predictions - torch.mean(predictions)
    vy = targets - torch.mean(targets)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return corr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNnet().to(device)

loss_fn = nn.MSELoss()
criterion_mae = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), 0.01)


data = np.load('data/6000_100.npy')

for epoch in range(200):
    
    model.train()
    train_loss_mse = 0
    train_loss_mae = 0
    train_loss_rmse = 0
    train_corr_coef = 0
    
    for i in range(4000):
        X = data[0+i:30+i]


        Y = data[30+i:60+i]
        X = np.expand_dims(X, axis=0)
        Y = np.expand_dims(Y, axis=0)


        X_tensor = torch.Tensor(X).to('cuda')
        Y_tensor = torch.Tensor(Y).to('cuda')
        output = model(X_tensor)

        loss = loss_fn(output, Y_tensor)
        
        loss_mse = loss_fn(output, Y_tensor)
        loss_mae = criterion_mae(output, Y_tensor)
        loss_rmse = rmse(output, Y_tensor)
        corr_coef = correlation_coefficient(output, Y_tensor)
        
        train_loss_mse += loss_mse.item()
        train_loss_mae += loss_mae.item()
        train_loss_rmse += loss_rmse.item()
        train_corr_coef += corr_coef.item()

        
        optimizer.zero_grad()

        
        loss.backward()


        optimizer.step()

    train_loss = train_loss_mse/4000

    print('Epoch [{}/200], MSE Loss: {}'.format(epoch+1, train_loss))
    print('MAE Loss:', train_loss_mae/4000)
    print('RMSE Loss:', train_loss_rmse/4000)
    print('Correlation Coefficient:', train_corr_coef/4000)


    if (epoch + 1) % 10 == 0:
        model.eval() 
        test_loss_mse = 0
        test_loss_mae = 0
        test_loss_rmse = 0
        test_corr_coef = 0
        for i in range(1500):
            X = data[4060+i:4090+i]


            Y = data[4090+i:4120+i]
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)


            X_tensor = torch.Tensor(X).to('cuda')
            Y_tensor = torch.Tensor(Y).to('cuda')
            output = model(X_tensor)
            

            loss_mse = loss_fn(output, Y_tensor)
            loss_mae = criterion_mae(output, Y_tensor)
            loss_rmse = rmse(output, Y_tensor)
            corr_coef = correlation_coefficient(output, Y_tensor)

            test_loss_mse += loss_mse.item()
            test_loss_mae += loss_mae.item()
            test_loss_rmse += loss_rmse.item()
            test_corr_coef += corr_coef.item()

        test_loss = test_loss_mse/1500

        print('-------------------------------------------')
        print(f'Epoch [{epoch+1}/200], test MSE Loss: {test_loss}')
        print('test MAE Loss:', test_loss_mae/1500)
        print('test RMSE Loss:', test_loss_rmse/1500)
        print('test Correlation Coefficient:', test_corr_coef/1500)
        print('-------------------------------------------')

torch.save(model.state_dict(), 'checkpoints/RNN_0.01.pth')