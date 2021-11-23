# Helper functions (will move to another file later)
import math

import torch
import torch.nn as nn

def deprocess(imgs):
    imgs = imgs * 255
    imgs = torch.clip(imgs,0,255)
    return imgs

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_pred)

# Pytorch doesn't support same padding with stride =/= 1, this is my fix
class PaddedConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super(PaddedConv2d, self).__init__()
        padding = (int(math.ceil((kernel_size[0]-1)/2)),
                   int(math.floor((kernel_size[0]-1)/2)),
                   int(math.ceil((kernel_size[1] - 1) / 2)),
                   int(math.floor((kernel_size[1] - 1) / 2)),
                   )
        self.net = nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        )
    def forward(self,x):
        return self.net(x)
