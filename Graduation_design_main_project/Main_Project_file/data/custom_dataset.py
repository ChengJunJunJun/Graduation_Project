from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

data = np.load('data/4000_1s_data_eta00.npy')
print(data.shape)
a = data[0:80,0,0]
print(a.shape)







# class Wave_Data(Dataset):
#     def __init__(self,) :
#         super(Wave_Data,self).__init__()
       

#     def __len__(self):
        
#         return 950
    
#     def __getitem__(self,index):
#         wave_data = np.load('../data/data/4000_1s_data_eta00.npy')
#         wave_data_tensor = torch.Tensor(wave_data)
#         tau = 50
#         features = torch.zeros((1000-50, 50, 360, 320))
#         labels = torch.zeros((1000-50, 50, 360, 320))
#         features.size()
#         for i in range(tau):
#             features[:,i] = wave_data_tensor[i:950+i]
#             labels[:,i] = wave_data_tensor[50+i:1000+i]
        
#         return features, labels
        



# if __name__ == "__main__":
#     train_data_custom = Wave_Data()
#     train_dataloader_custom = DataLoader(dataset=train_data_custom, # use custom created train Dataset
#                                      batch_size=32, # how many samples per batch?
#                                      num_workers=0, # how many subprocesses to use for data loading? (higher = more)
#                                      shuffle=True) # shuffle the data?


#     pass

