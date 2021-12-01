import torch.nn
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
    for i in range(23,31):
        vgg16.features[i] = Identity()
    return vgg16, torch.nn.Unflatten(dim=1,unflattened_size=(512,28,28))


def get_vgg_top():
    # Expects images normalized via
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    vgg16 = models.vgg16(pretrained=True)
    # Strip top layers
    for i in range(0, 23):
        vgg16.features[i] = Identity()
    return vgg16


if __name__ == '__main__':
    bot, _ = get_partial_vgg()
    print(bot)
    print(get_vgg_top())
