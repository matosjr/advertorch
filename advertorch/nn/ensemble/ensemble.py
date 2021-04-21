from torch import nn

from advertorch.nn.normalization.normalize import Normalize


class EnsembleModule(nn.Module):
    def __init__(self, normalize: Normalize, nets: list):
        super(EnsembleModule, self).__init__()
        self.normalize = normalize
        self.nets = nn.ModuleList([net for net in nets])

    def forward(self, x):
        x = self.normalize(x)
        outputs = [net(x) for net in self.nets]
        return outputs
