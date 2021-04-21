import torch

import numpy as np

from advertorch.attacks import Attack, LabelMixin


def l2_normalize(x_t):
    x = x_t.clone()
    return x / (x.square().mean().sqrt() + 1e-5)


class _DeepXploreMethod(Attack, LabelMixin):
    def __init__(self, predict,
                 loss_fn,
                 step,
                 grad_iterations,
                 targeted):
        super(_DeepXploreMethod, self).__init__(predict=predict,
                                                loss_fn=loss_fn,
                                                clip_min=None,
                                                clip_max=None)

        self.__targeted = targeted
        self.step = step
        self.grad_iterations = grad_iterations

    @property
    def targeted(self):
        return self.__targeted

    def perturb(self, x, y):
        x, y = self._verify_and_process_inputs(x, y)
        delta = torch.zeros_like(x)
        delta.requires_grad = True

        for i in range(self.grad_iterations):
            self.predict.zero_grad()
            outputs = self.predict(x + delta)
            # TODO: Add break clause
            cost = self.loss_fn(outputs, y)
            cost.backward()

            grad = delta.grad.data
            grad = l2_normalize(grad)
            grad = self.transformation(grad)
            grad = grad * self.step
            delta.data = delta.data + grad
            delta.data = delta.data.clamp(-1., 1.)
            delta.grad.data.zero_()
        x = x + delta
        x = x.clamp(0., 1.)
        return x

    def transformation(self, gradients):
        return gradients


class DeepXploreBlackout(_DeepXploreMethod):
    def __init__(self,
                 predict,
                 loss_fn,
                 step,
                 grad_iterations,
                 targeted,
                 rect_shape=(6, 6)):
        super(DeepXploreBlackout, self).__init__(
            predict, loss_fn, step, grad_iterations, targeted)
        self.rect_shape = rect_shape

    def transformation(self, gradients):
        start_point = (
            np.random.randint(0, gradients.shape[-2] - self.rect_shape[0]),
            np.random.randint(0, gradients.shape[-1] - self.rect_shape[1])
        )

        new_grads = torch.zeros_like(gradients)
        patch = gradients.clone()[
                ...,
                start_point[0]:start_point[0] + self.rect_shape[0],
                start_point[1]:start_point[1] + self.rect_shape[1]
                ]

        if torch.mean(patch) < 0:
            new_grads[
            ...,
            start_point[0]:start_point[0] + self.rect_shape[0],
            start_point[1]:start_point[1] + self.rect_shape[1]] = -torch.ones_like(patch)
        return new_grads


class DeepXploreLight(_DeepXploreMethod):
    def __init__(self,
                 predict,
                 loss_fn,
                 step,
                 grad_iterations,
                 targeted):
        super(DeepXploreLight, self).__init__(
            predict, loss_fn, step, grad_iterations,targeted
        )

    def transformation(self, gradients):
        new_grads = torch.ones_like(gradients)
        grad_mean = gradients.clone().detach().mean()
        return grad_mean * new_grads


class DeepXploreOcclusion(_DeepXploreMethod):
    def __init__(self,
                 predict,
                 loss_fn,
                 step,
                 grad_iterations,
                 targeted,
                 starting_point,
                 occlusion_size):
        super(DeepXploreOcclusion, self).__init__(predict,
                                                  loss_fn,
                                                  step,
                                                  grad_iterations,
                                                  targeted)
        self.start_point = starting_point
        self.rect_shape = occlusion_size

    def transformation(self, gradients):
        new_grads = torch.zeros_like(gradients)
        new_grads[
        ...,
        self.start_point[0]:self.start_point[0] + self.rect_shape[0],
        self.start_point[1]:self.start_point[1] + self.rect_shape[1]
        ] = gradients.clone()[
            ...,
            self.start_point[0]:self.start_point[0] + self.rect_shape[0],
            self.start_point[1]:self.start_point[1] + self.rect_shape[1]
            ]
        return new_grads


DXBlackout = DeepXploreBlackout
DXLight = DeepXploreLight
DXOcclusion = DeepXploreOcclusion
