import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from datetime import datetime
from torch.utils.data import DataLoader
import torchvision.models as models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def get_partial_vgg():
    # Expects images normalized via
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    vgg16 = models.vgg16(pretrained=True)
    # Strip top layers
    vgg16.avgpool = Identity()
    vgg16.classifier = Identity()
    return vgg16
