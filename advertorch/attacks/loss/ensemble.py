from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules.loss import _Loss


# class NeuronLossParameters(_Loss):
#
#     def _forward_unimplemented(self, *input: Any) -> None:
#         pass
#     def __init__(self, neuron_data: CoverageData, x: torch.Tensor, y: torch.Tensor):
#         super(NeuronLossParameters, self).__init__(x, y)
#         self.neuron_data = neuron_data


def _hidden_loss(i, neuron_data, hidden_outputs):
    layer_name, idx = neuron_data
    outputs = hidden_outputs[i][layer_name][:, idx, ...]
    return F.relu(outputs).mean()


class Loss(_Loss):

    def _forward_unimplemented(self, *input_: Any) -> None:
        error = "Sub-classes must implement forward."
        raise NotImplementedError(error)

    def forward(self, input_: Tensor, target: Tensor):
        error = "Sub-classes must implement forward."
        raise NotImplementedError(error)


class EnsembleCrossEntropyLoss(Loss):
    def __init__(self, loss_factors: torch.DoubleTensor):
        super(EnsembleCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.loss_factors = loss_factors

    def forward(self, input_: Tensor, target: Tensor):
        loss = [self.loss_fn(x, target) for x in input_]
        loss = torch.stack(loss, dim=0)
        return loss * self.loss_factors


class ReversedLossFunctionEnsemble(EnsembleCrossEntropyLoss):
    def forward(self, input_: Tensor, target: Tensor):
        loss = [self.loss_fn(x, target) for x in input_]
        loss = torch.stack(loss, dim=0)
        loss = loss * self.loss_factors
        return loss.sum()


class NeuronCoverageLoss(Loss):
    def __init__(self, neuron_data, weigh_nc):
        super(NeuronCoverageLoss, self).__init__()
        self.__neuron_data = neuron_data
        self.__weigh_nc = weigh_nc

    def forward(self):
        losses = [_hidden_loss(i, neuron_data, self.__neuron_data.hidden_outputs)
                  for i, neuron_data in enumerate(self.__neuron_data.neurons_to_cover)]
        losses = torch.stack(losses)
        return self.__weigh_nc * losses.sum()


class DeepXploreCrossEntropyLoss(Loss):
    def __init__(self, ensemble: EnsembleCrossEntropyLoss, neuron: NeuronCoverageLoss, targeted: bool):
        super(DeepXploreCrossEntropyLoss, self).__init__()
        self.__signal = -1.0 if targeted else 1.0
        self.__ensemble = ensemble
        self.__neuron = neuron

    def forward(self, input_: Tensor, target: Tensor):
        loss = self.__ensemble(input_, target).sum()
        loss_nc = self.__neuron()
        return self.__signal * (loss + loss_nc)


RLFE = ReversedLossFunctionEnsemble
NCLoss = NeuronCoverageLoss
DXLoss = DeepXploreCrossEntropyLoss
