import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """Class definition for LeNet5"""

    def __init__(self, input_channels=1, num_classes=10):
        super(LeNet5, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # 1 input image channel, 6 output channels, 5x5 square conv kernel, padding of 2
        self.conv1 = nn.Conv2d(in_channels=self.input_channels,
                               out_channels=6,
                               kernel_size=(5, 5),
                               padding=2)
        self.pool1 = nn.MaxPool2d((2, 2))
        # 6 input image channel, 16 output channels, 5x5 square conv kernel, padding of 2
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=(5, 5),
                               padding=2)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.flatten = nn.Flatten()

        # 7x7 image dimension
        self.fc1 = nn.Linear(in_features=16 * 7 * 7,
                             out_features=120)

        self.fc2 = nn.Linear(in_features=120, out_features=84)

        self.fc3 = nn.Linear(in_features=84, out_features=self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x
