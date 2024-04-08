import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.crnn import GRUnet as GRUnet


model = GRUnet().to('cuda')
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 0.001)

eta00 = np.load('data/6000_1s_data_eta00.npy')
# 选取第20个角度的第100到200个所有的时序数据；
data = eta00[:,20,100:200]

for epoch in range(200):
    
    model.train()
    train_loss = 0
    
    for i in range(4000):
        X = data[0+i:30+i]


        Y = data[30+i:60+i]
        X = np.expand_dims(X, axis=0)
        Y = np.expand_dims(Y, axis=0)


        X_tensor = torch.Tensor(X).to('cuda')
        Y_tensor = torch.Tensor(Y).to('cuda')
        output = model(X_tensor)

        loss = loss_fn(output, Y_tensor)
        train_loss += loss.item() 

        
        optimizer.zero_grad()

        
        loss.backward()


        optimizer.step()

    train_loss =  train_loss / 4000

    print(f'Epoch [{epoch+1}/200], train Loss: {train_loss}')


    if (epoch + 1) % 10 == 0:
        model.eval() 
        test_loss = 0
        for i in range(1500):
            X = data[4060+i:4090+i]


            Y = data[4090+i:4120+i]
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)


            X_tensor = torch.Tensor(X).to('cuda')
            Y_tensor = torch.Tensor(Y).to('cuda')
            output = model(X_tensor)
            
            loss = loss_fn(output, Y_tensor)

            test_loss += loss.item()
        test_loss = test_loss / 1500

        print('-------------------------------------------')
        print(f'Epoch [{epoch+1}/200], test Loss: {test_loss}')
        print('-------------------------------------------')

torch.save(model.state_dict(), 'checkpoints/GRUnet_model_parameters.pth_0.001')