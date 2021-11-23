import collections
import math

import torch
from torch import nn

# Hyperparams (move to text file later)
import partial_vgg
import train_vgg


class Configuration():
    def __init__(self):
        self.batch_size = 64
        self.epochs = 5
        self.IMAGE_SIZE = 224
        self.nThreads = 4
        self.learning_rate = 0.01


config = Configuration()


def get_pretrained_partial_vgg():
    model = partial_vgg.VGG16(full_model=False)
    pretrained_dict = torch.load('weights/vgg.pkl')
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    pretrained_dict = [(k, v) for k, v in pretrained_dict.items()]
    pretrained_dict = collections.OrderedDict(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return model


def train_gan():
    train_set, test_set = train_vgg.load_imagenet(config.batch_size, config.nThreads)
    vgg = get_pretrained_partial_vgg()