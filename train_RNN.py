import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.crnn import RNNnet as RNNnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNnet().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 0.001)

data = np.load('data/chengjun.npy')



for epoch in range(200):
    
    model.train()
    train_loss = 0
    
    for i in range(40000):
        X = data[0+i:30+i]


        Y = data[30+i:60+i]
        X = np.expand_dims(X, axis=0)
        Y = np.expand_dims(Y, axis=0)


        X_tensor = torch.Tensor(X).to(device)
        Y_tensor = torch.Tensor(Y).to(device)
        output = model(X_tensor)

        loss = loss_fn(output, Y_tensor)
        train_loss += loss.item() 

        
        optimizer.zero_grad()

        
        loss.backward()


        optimizer.step()

    train_loss =  train_loss / 40000

    print(f'Epoch [{epoch+1}/200], train Loss: {train_loss}')


    if epoch % 10 == 0:
        model.eval() 
        test_loss = 0
        for i in range(15000):
            X = data[40060+i:40090+i]


            Y = data[40090+i:40120+i]
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)


            X_tensor = torch.Tensor(X).to(device)
            Y_tensor = torch.Tensor(Y).to(device)
            output = model(X_tensor)
            
            loss = loss_fn(output, Y_tensor)

            test_loss += loss.item()
        test_loss = test_loss / 15000

        print('-------------------------------------------')
        print(f'Epoch [{epoch+1}/200], test Loss: {test_loss}')
        print('-------------------------------------------')

        
torch.save(model.state_dict(), 'checkpoints/RNN_0.001.pth')