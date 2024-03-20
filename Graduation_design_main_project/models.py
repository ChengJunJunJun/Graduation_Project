import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch
import torch.optim as optim


class WaveNet(nn.Module):
    def __init__(self, cnn_channels, rnn_hidden_size, rnn_layers):
        super(WaveNet, self).__init__()
        self.cnn_channels = cnn_channels
        self.rnn_hidden_size = rnn_hidden_size # 252
        self.rnn_layers = rnn_layers

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=cnn_channels, out_channels=1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # RNN layers
        self.rnn = nn.GRU(input_size=57600, hidden_size=rnn_hidden_size, num_layers=rnn_layers, batch_first=True)

        # # CNN layers after RNN
        # self.conv3 = nn.Conv1d(in_channels=rnn_hidden_size, out_channels=cnn_channels, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1)

        # Output layer
        self.fc = nn.Linear(in_features=rnn_hidden_size, out_features=230400)

    def forward(self, x):
        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        print(x.size())

        # Reshape for RNN
        x = x.unsqueeze(0) # Change shape from [batch_size, channels, sequence_length] to [batch_size, sequence_length, channels]

        # RNN layers
        x, _ = self.rnn(x)


        # Output layer
        x = self.fc(x)

        x = x.view(80,720,320)
        return x
def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列

    Defined in :numref:`sec_language_model`"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 30) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)
    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 30) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

# 读取.npy文件
eta00 = np.load('data/1000_1s_data_eta00.npy')
model = WaveNet(64, 256, 1).to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), 0.01)


for X, Y in seq_data_iter_random(eta00, batch_size=1, num_steps=80):
    X = X.permute(1, 0, 2, 3).float().to('cuda')
    output = model(X)
    # output, hidden_prev = model(X, hidden_prev)
    # hidden_prev = hidden_prev.detach()
    Y = Y.squeeze().float().to('cuda')
    loss = criterion(output, Y)
    model.zero_grad()
    loss.backward()

    optimizer.step()


    print("Iteration:{} loss {}".format(iter, loss.item()))