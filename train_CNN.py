import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.crnn import CNNnet as CNNnet


model = CNNnet().to('cuda')
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 0.001)

data = np.load('data/chengjun.npy')


for epoch in range(200):
    
    model.train()
    train_loss = 0
    
    for i in range(4000):
        X = data[0+i:30+i]


        Y = data[30+i:52+i]
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

    print(f'Epoch [{epoch+1}/200], train Loss: {train_loss}')


    if (epoch + 1) % 10 == 0:
        model.eval() 
        test_loss = 0
        for i in range(1500):
            X = data[4060+i:4090+i]


            Y = data[4090+i:4112+i]
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)


            X_tensor = torch.Tensor(X).to('cuda').permute(0,2,1)
            Y_tensor = torch.Tensor(Y).to('cuda')
            output = model(X_tensor)
            
            loss = loss_fn(output, Y_tensor)

            test_loss += loss.item()
        test_loss = test_loss / 1500

        print('-------------------------------------------')
        print(f'Epoch [{epoch+1}/200], test Loss: {test_loss}')
        print('-------------------------------------------')

torch.save(model.state_dict(), 'checkpoints/CNNnet_model_parameters_0.001.pth')