import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, input_dims=(3, 224, 224)) -> None:
        super(SimpleCNN, self).__init__()

        in_channels = input_dims[0]

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PaperCnn(nn.Module):
    """Copied Model from Paper"""

    def __init__(self, input_dims=(3, 224, 224)) -> None:
        super(PaperCnn, self).__init__()
        in_channels = input_dims[0]

        self.elu = nn.ELU()
        self.dropout = nn.Dropout()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=24, kernel_size=5, stride=2
        ) #110
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2) #53
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2) #25
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3) #23
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3) # 21

        self.fc1 = nn.Linear(64*21*21, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.elu(self.conv5(x))
        # x = self.dropout(x)
        x = x.view( x.size(0),-1)
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.fc4(x)
        return x
