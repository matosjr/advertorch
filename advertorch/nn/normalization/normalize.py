from abc import ABC

import torch
import torch.nn as nn


class Normalize(nn.Module, ABC):
    """description of class"""

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        mean = self.mean.type_as(x)[None, :, None, None]
        std = self.std.type_as(x)[None, :, None, None]
        return (x - mean) / std
