import torch.nn as nn
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


def get_vgg_top():
    # Expects images normalized via
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    vgg16 = models.vgg16(pretrained=True)
    # Strip top layers
    vgg16.features = Identity()
    return vgg16


if __name__ == '__main__':
    print(get_partial_vgg())
    print(get_vgg_top())
