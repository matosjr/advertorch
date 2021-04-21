import torch

from nta.nn.lenet.lenet import LeNet
from nta.nn.lenet.lenet4 import LeNet4
from nta.nn.lenet.lenet5 import LeNet5


def _load_parameters(module, pretrained, parameters_filename=None):
    module_ = module()
    if pretrained:
        assert parameters_filename, 'Must provide parameters file for pretrained model.'
        parameters_filename
        module_.load_state_dict(torch.load(parameters_filename))
    return module_


def lenet(pretrained, parameters_filename=None):
    return _load_parameters(LeNet, pretrained, parameters_filename)


def lenet4(pretrained, parameters_filename=None):
    return _load_parameters(LeNet4, pretrained, parameters_filename)


def lenet5(pretrained, parameters_filename=None):
    return _load_parameters(LeNet5, pretrained, parameters_filename)
