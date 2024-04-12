import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.gru import GruNet as GruNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GruNet(input_size=1, hidden_size=16, output_size=1).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 0.001)

data = np.load('data/chengjun_new.npy')

for epoch in range(200):
    
    model.train()
    train_loss = 0
    
    for i in range(131):
        X = data[0+i*100:1000+i*100]


        Y = data[1000+i*100:2000+i*100]
        X = np.expand_dims(X, axis=2)
        Y = np.expand_dims(Y, axis=2)


        X_tensor = torch.Tensor(X).to(device)
        Y_tensor = torch.Tensor(Y).to(device)
        output = model(X_tensor)

        loss = loss_fn(output, Y_tensor)
        train_loss += loss.item() 

        
        optimizer.zero_grad()

        
        loss.backward()


        optimizer.step()

    train_loss =  train_loss / 131

    print(f'Epoch [{epoch+1}/200], train Loss: {train_loss}')


    if (epoch + 1) % 10 == 0:
        model.eval() 
        test_loss = 0
        for i in range(31):
            X = data[15000+i*100:16000+i*100]


            Y = data[16000+i*100:17000+i*100]
            X = np.expand_dims(X, axis=2)
            Y = np.expand_dims(Y, axis=2)


            X_tensor = torch.Tensor(X).to(device)
            Y_tensor = torch.Tensor(Y).to(device)
            output = model(X_tensor)
            
            loss = loss_fn(output, Y_tensor)

            test_loss += loss.item()
        test_loss = test_loss / 31

        print('-------------------------------------------')
        print(f'Epoch [{epoch+1}/200], test Loss: {test_loss}')
        print('-------------------------------------------')

torch.save(model.state_dict(), 'checkpoints/CRNN_0.001.pth')