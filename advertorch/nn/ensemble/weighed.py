import torch
import torch.nn as nn

from .ensemble import EnsembleModule


class WeighedEnsembleModule(EnsembleModule):
    def __init__(self, nets: list):
        super(WeighedEnsembleModule, self).__init__(nets)

    def _forward(self, outputs):
        # output = torch.cat(outputs, dim=0)
        # output = output.sum(dim=1)
        return outputs


if __name__ == "__main__":
    def create_module(v):
        module = nn.Linear(10, 10)
        module.weight.data.zero_()
        module.weight.data = module.weight.data.zero_() + v
        module.bias.data = module.bias.data.zero_() + v
        return module


    module_list = WeighedEnsembleModule([create_module(i) for i in range(5)])
    module_list.eval()

    x = torch.ones(1, 10)
    print(module_list(x))
    print(module_list.training)

    for name, module in module_list.named_modules():
        print(name, type(module))
