import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.crnn import CRNNnet as CRNNnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNNnet().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 0.001)

data = np.load('data/chengjun_new.npy')



for epoch in range(200):
    
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


    if (epoch + 1) % 10 == 0:
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

torch.save(model.state_dict(), 'checkpoints/CRNNnet_model_parameters.pth')