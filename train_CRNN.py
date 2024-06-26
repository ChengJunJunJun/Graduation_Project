import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.crnn import CRNNnet , CLSTMnet
import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="CRNN_pridict_waveheight",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "CRNN",
    "dataset": "6000_100.npy",
    "epochs": 200,
    }
)
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
model = CRNNnet().to(device)

loss_fn = nn.MSELoss()
criterion_mae = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), 0.001)

data = np.load('data/6000_100.npy')



for epoch in range(200):
    
    model.train()
    train_loss_mse = 0
    train_loss_mae = 0
    train_loss_rmse = 0
    train_corr_coef = 0
    
    for i in range(4000):
        X = data[0+i:60+i]


        Y = data[60+i:120+i]
        X = np.expand_dims(X, axis=0)
        Y = np.expand_dims(Y, axis=0)


        X_tensor = torch.Tensor(X).to('cuda').permute(0,2,1)
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

    # log metrics to wandb
    wandb.log({"train MSE Loss": train_loss})



    if (epoch + 1) % 4 == 0:
        model.eval() 
        test_loss_mse = 0
        test_loss_mae = 0
        test_loss_rmse = 0
        test_corr_coef = 0
        for i in range(1500):
            X = data[4120+i:4180+i]


            Y = data[4180+i:4240+i]
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)


            X_tensor = torch.Tensor(X).to('cuda').permute(0,2,1)
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
        wandb.log({"test MSE Loss": test_loss})

torch.save(model.state_dict(), 'checkpoints/CRNN_0.01_60.pth')

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()