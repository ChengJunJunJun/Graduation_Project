import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__() #调用了父类 nn.Module 的构造函数

        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=16,
            num_layers=1,
            batch_first=False,
        )
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p, mean=0.0, std=0.001)   # 对参数进行初值化

        self.linear = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.rnn(x)  # [b, seq, h]
        out = self.linear(out)
        return out
    
import torch.optim as optim

model = Net().to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 0.01)
model.load_state_dict(torch.load('model_parameters.pth'))
eta00 = np.load('Main_Project_file/data/data/4000_1s_data_eta00.npy')
data = eta00[:,0]

X = data[0]
X_tensor = torch.Tensor(X).unsqueeze(-1).to('cuda')
output = model(X_tensor)
print(output.size())

a = output.cpu().view(320).detach().numpy()


# 选择第一帧数据和第一个角度
frame_index = 4
angle_index = 0
eta_frame = eta00[frame_index]
eta_angle = eta_frame[angle_index]

# 设置径向范围
r_values = np.linspace(0, 3200, 320)

# 可视化波浪形状
plt.figure(figsize=(10, 6))
plt.plot(r_values, eta_angle)
plt.plot(r_values, a)
plt.title('Wave Shape at Angle 0')
plt.xlabel('Radial Distance')
plt.ylabel('Wave Height')
plt.grid(True)
plt.show()