import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveNet(nn.Module):
    def __init__(self, cnn_channels, rnn_hidden_size, rnn_layers, num_classes):
        super(WaveNet, self).__init__()
        self.cnn_channels = cnn_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.num_classes = num_classes

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # RNN layers
        self.rnn = nn.GRU(input_size=cnn_channels, hidden_size=rnn_hidden_size, num_layers=rnn_layers, batch_first=True)

        # CNN layers after RNN
        self.conv3 = nn.Conv1d(in_channels=rnn_hidden_size, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1)

        # Output layer
        self.fc = nn.Linear(in_features=cnn_channels, out_features=num_classes)

    def forward(self, x):
        # CNN layers
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        # Reshape for RNN
        x = x.permute(0, 2,
                      1)  # Change shape from [batch_size, channels, sequence_length] to [batch_size, sequence_length, channels]

        # RNN layers
        x, _ = self.rnn(x)

        # CNN layers after RNN
        x = F.relu(self.conv3(x.permute(0, 2, 1)))  # Change shape back to [batch_size, channels, sequence_length]
        x = self.pool(F.relu(self.conv4(x)))

        # Global average pooling
        x = F.avg_pool1d(x, kernel_size=x.shape[2])
        x = x.view(-1, self.cnn_channels)

        # Output layer
        x = self.fc(x)
        return x


# Example usage:
# Define model parameters
cnn_channels = 64
rnn_hidden_size = 128
rnn_layers = 2
num_classes = 10

# Create model instance
model = WaveNet(cnn_channels, rnn_hidden_size, rnn_layers, num_classes)

# Example input
batch_size = 32
sequence_length = 100
input_channels = 1
input_data = torch.randn(batch_size, input_channels, sequence_length)

# Forward pass
output = model(input_data)
print("Output shape:", output.shape)
