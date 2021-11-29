# Helper functions (will move to another file later)
import math

import numpy
import numpy as np
import torch
import torch.nn as nn

def get_all_dims(tensor):
    return tuple(range(0, len(tensor.shape)))

def random_weighted_average(a, b, device='cpu'):
    batch = a.shape[0]
    weights = torch.rand(batch).to(device)
    return weights[:, None, None, None] * a + (1 - weights[:, None, None, None]) * b


def deprocess(imgs):
    imgs = imgs * 255
    imgs = torch.clip(imgs, 0, 255)
    return imgs


def wasserstein_loss(y_pred):
    return torch.mean(y_pred)


def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


def gradient_penalty_loss(y_pred, averaged_samples,
                          gradient_penalty_weight=10):
    # Intent:
    # Get gradient from y_pred down to averaged_samples
    # norm = sqrt(sum_all_but_batch(grads**2))
    # Penalty = mean_over_batch(weight*(1-norm)**2)
    ones = torch.tensor(numpy.ones_like(y_pred.detach().numpy()))
    gradients = torch.autograd.grad(y_pred, averaged_samples, grad_outputs=ones,retain_graph=True)[0]

    gradients_sqr = gradients ** 2
    gradients_sqr_sum = torch.sum(gradients_sqr,
                                  dim=get_all_dims(y_pred)[1:])
    gradient_l2_norm = torch.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * (1 - gradient_l2_norm) ** 2
    return torch.mean(gradient_penalty)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.autograd.Variable(torch.Tensor(d_interpolates.shape).fill_(1.0), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Pytorch doesn't support 'same' padding with stride =/= 1, this is my fix
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
