# import torch.nn as nn
#
#
# class LossParameters:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
#
# class BaseLoss(nn.Module):
#     def forward(self, params: LossParameters):
#         return self._forward(params)
#
#     def _forward_unimplemented(self, *input_) -> None:
#         pass
#
#     def _forward(self, params: LossParameters):
#         raise NotImplementedError("Base class!!")
