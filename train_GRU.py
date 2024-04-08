import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.crnn import GRUnet as GRUnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUnet().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 0.001)

data = np.load('data/chengjun.npy')

for epoch in range(200):
    
    model.train()
    train_loss = 0
    
    for i in range(800):
        X = data[i*60:i*60+30]


        Y = data[i*60+30:i*60+60]
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

    train_loss =  train_loss / 800

    print(f'Epoch [{epoch+1}/200], train Loss: {train_loss}')


    if (epoch + 1) % 10 == 0:
        model.eval() 
        test_loss = 0
        for i in range(200):
            X = data[48000 + (i*60):48000 + (i*60) +30]


            Y = data[48000 + (i*60) +30:48000 + (i*60) +60]
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)


            X_tensor = torch.Tensor(X).to(device)
            Y_tensor = torch.Tensor(Y).to(device)
            output = model(X_tensor)
            
            loss = loss_fn(output, Y_tensor)

            test_loss += loss.item()
        test_loss = test_loss / 200

        print('-------------------------------------------')
        print(f'Epoch [{epoch+1}/200], test Loss: {test_loss}')
        print('-------------------------------------------')

torch.save(model.state_dict(), 'checkpoints/GRU_0.001.pth')