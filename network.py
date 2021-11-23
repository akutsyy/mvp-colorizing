import math

import torch
from torch import nn


# Hyperparams (move to text file later)

class Configuration():
    def __init__(self):
        self.batch_size = 10
        self.IMAGE_SIZE = 224
        self.nThreads = 4

config = Configuration()

# Helper functions (will move to another file later)
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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            PaddedConv2d(in_channels=(1+2),out_channels=64,kernel_size=(4,4),stride=(2,2)),
            nn.LeakyReLU(),
            PaddedConv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=(2,2)),
            nn.LeakyReLU(),
            PaddedConv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2)),
            nn.LeakyReLU(),
            PaddedConv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2)),
            nn.LeakyReLU(),
            PaddedConv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=(2, 2)), # 1,28,28
        )
    def forward(self, x):
        # x is shape (1+2,img_size,img_size)
        return self.net(x)

class Colorization_Model(nn.Module):
    def __init__(self):
        super(Colorization_Model, self).__init__()

        # Following the breakdown in ChromaGAN
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()

        self.global_features = nn.Sequential(
            PaddedConv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(2,2)),
            self.activation,
            nn.BatchNorm2d(512),
            PaddedConv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1)),
            self.activation,
            nn.BatchNorm2d(512),
            PaddedConv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2)),
            self.activation,
            nn.BatchNorm2d(512),
            PaddedConv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1)),
            self.activation,
            nn.BatchNorm2d(512),
        )

        # ChromaGAN source code doesn't have activations in the dense layers,
        # I've added ReLU since that seems like an error
        self.global_features2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4,1024),
            self.activation,
            nn.Linear(1024,512),
            self.activation,
            nn.Linear(512,256),
            self.activation,
            # Repeating, reshaping done in forward
        )

        self.global_features_class = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4, 4096),
            self.activation,
            nn.Linear(4096, 4096),
            self.activation,
            nn.Linear(4096, 1000),
            nn.Softmax(),
        )

        # Midlevel feature
        self.midlevel_features = nn.Sequential(
            nn.Flatten(),
            PaddedConv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=(1,1)),
            self.activation,
            nn.BatchNorm2d(512),
            PaddedConv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            self.activation,
            nn.BatchNorm2d(256),
        )

        self.output_model = nn.Sequential(
            PaddedConv2d(in_channels=512,out_channels=256,kernel_size=(1, 1),stride=(1,1)),
            self.activation,
            PaddedConv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            self.activation,

            nn.Upsample(scale_factor=2,mode='nearest'),
            PaddedConv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            self.activation,
            PaddedConv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            self.activation,

            nn.Upsample(scale_factor=2, mode='nearest'),
            PaddedConv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            self.activation,
            PaddedConv2d(in_channels=32, out_channels=2, kernel_size=(3, 3), stride=(1, 1)),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2,mode='nearest')

        )

    def forward(self, x):
        # Output is shape BATCH,512,14,14 and comes from VGG16

        batch_size = x.shape[0]

        globalFeatures = self.global_features(x)

        globalFeatures2 = self.global_features2(globalFeatures)
        #TODO: Check that this is correct (tensorflow and pytorch have channels in different dimensions)
        globalFeatures2 = globalFeatures2.unsqueeze(1).repeat(1,28*28,1)
        globalFeatures2 = torch.reshape(globalFeatures2,(batch_size,28,28,256))
        globalFeatures2 = torch.permute(globalFeatures2,(0,3,1,2))

        globalFeaturesClass = self.global_features_class(globalFeatures)

        midlevelFeatures = self.midlevel_features(x)

        # Fuse (VGG16+Midlevel), (VGG16+Global)
        modelFusion = torch.cat([midlevelFeatures,globalFeatures2],dim=1)

        outputModel = self.output_model(modelFusion)

        return outputModel, globalFeaturesClass









