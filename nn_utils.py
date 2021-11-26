# Helper functions (will move to another file later)
import math

import numpy as np
import torch
import torch.nn as nn


def random_weighted_average(a, b):
    batch = a.shape[0]
    weights = torch.rand(batch)
    return weights[:, None, None, None] * a + (1 - weights[:, None, None, None]) * b


def deprocess(imgs):
    imgs = imgs * 255
    imgs = torch.clip(imgs, 0, 255)
    return imgs


def wasserstein_loss(y_pred, y_true):
    return torch.mean(y_pred,dim=(0,1,2,3))


def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2,dim=(0,1,2,3))


def gradient_penalty_loss(y_pred, y_true, averaged_samples,
                          gradient_penalty_weight=10):
    y_pred.zero_grad()
    y_pred.backward()
    gradients = averaged_samples.grad
    print(gradients.shape)
    gradients_sqr = gradients ** 2
    gradients_sqr_sum = torch.sum(gradients_sqr,
                                  dim=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = gradients_sqr_sum ** 2
    gradient_penalty = gradient_penalty_weight * (1 - gradient_l2_norm) ** 2
    print(gradient_penalty.shape)
    return torch.mean(gradient_penalty)


# Pytorch doesn't support same padding with stride =/= 1, this is my fix
class PaddedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(PaddedConv2d, self).__init__()
        padding = (int(math.ceil((kernel_size[0] - 1) / 2)),
                   int(math.floor((kernel_size[0] - 1) / 2)),
                   int(math.ceil((kernel_size[1] - 1) / 2)),
                   int(math.floor((kernel_size[1] - 1) / 2)),
                   )
        self.net = nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        )

    def forward(self, x):
        return self.net(x)
