import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """Class definition for LeNet"""

    def __init__(self, input_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # 1 input image channel, 4 output channels, 5x5 square conv kernel
        self.conv1 = nn.Conv2d(in_channels=self.input_channels,
                               out_channels=4,
                               kernel_size=(5, 5),
                               padding=2)

        self.conv2 = nn.Conv2d(in_channels=4,
                               out_channels=12,
                               kernel_size=(5, 5),
                               padding=2)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(in_features=12 * 7 * 7,
                            out_features=self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.flatten(x)
        x = self.fc(x)

        return x
