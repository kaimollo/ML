# 创建模型
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.maxpool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 62 * 62, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 2)
#
#     def forward(self, x):
#         x = self.maxpool(F.relu(self.conv1(x)))
#         x = self.maxpool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 62 * 62)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#
#         return x
import torch
from torch import nn
from torch.nn import ReLU, Sigmoid, Flatten


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 7),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.Conv2d(64, 256, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(256 *10 * 10, 1024),
            nn.ReLU(),
            torch.nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.ReLU(),
            torch.nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x
