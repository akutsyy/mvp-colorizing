import torch
import os
import random
from pathlib import Path

import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from skimage import color as skcolor
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

# Hyperparams (move to text file later)
import config


# Used 0.05 to generate dataset
def convert_videos_to_images(eval_portion=0.05):
    Path("dataset/UCF101Images_train").mkdir(parents=True, exist_ok=True)
    Path("dataset/UCF101Images_eval").mkdir(parents=True, exist_ok=True)
    total = len(os.listdir("dataset/UCF101"))
    for i, filename in enumerate(os.listdir("dataset/UCF101")):
        trimmed_filename = filename[:-4]
        print(str(100 * i / total) + "%,  " + filename)
        # Skip if already processed this video
        if not filename.endswith(".avi") or \
                os.path.isfile("dataset/UCF101Images_train/" + str(trimmed_filename) + "_0.png") \
                or os.path.isfile("dataset/UCF101Images_eval/" + str(trimmed_filename) + "_0.png"):
            continue

        frames, audio, metadata = torchvision.io.read_video("dataset/UCF101/" + filename)
        # Place entire videos either as eval or train, to avoid over-fitting
        do_eval = random.random() < eval_portion
        # Reduce training set to 2 fps to save space/time
        if not do_eval:
            frames = frames[::12]

        for i, img in enumerate(frames):
            tensor = cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_BGR2RGB)
            if do_eval:
                cv2.imwrite("dataset/UCF101Images_eval/" + str(trimmed_filename) + "_" + str(i) + ".png", tensor)
            else:
                cv2.imwrite("dataset/UCF101Images_train/" + str(trimmed_filename) + "_" + str(i) + ".png", tensor)


# Use the UCF101 dataset, here https://www.crcv.ucf.edu/data/UCF101.php
# or here: http://www.thumos.info/download.html
# This is a collection of human actions, chosen because many black and white videos
# focus on people (old movies, etc)

# Transform is required to output a tensor
class UCF101ImageDataset(Dataset):
    """UCF101 Dataset."""

    def __init__(self, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.filenames = os.listdir(root_dir)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.other_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.filenames[idx])
        transforms.ToTensor(),

        image = self.to_pil(torchvision.io.read_image(img_name))

        image = self.other_transforms(image)

        image = self.to_tensor(skcolor.rgb2lab(image))
        image = self.normalize(image)

        sample = [image[1:], image[0].unsqueeze(0)]

        return sample


def get_datasets():
    train_set = UCF101ImageDataset("dataset/UCF101Images_train")
    test_set = UCF101ImageDataset("dataset/UCF101Images_eval")
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=config.num_workers)
    return train_loader, test_loader, len(train_set), len(test_set)


def display_dataset_sample():
    # Used for viewing normalized images
    class UnNormalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            """
            Args:
                tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            Returns:
                Tensor: Normalized image.
            """
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
            return tensor

    training_data = UCF101ImageDataset("dataset/UCF101Images_train")
    rows = 6
    figure = plt.figure(figsize=(3, 6))
    figure.set_tight_layout(True)
    unnormalize = UnNormalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    for i in range(1, 12, 2):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        color, bw = training_data[sample_idx]
        full_image = torch.concat([bw, color], dim=0)
        full_image = unnormalize(full_image)
        bw = full_image[0]

        figure.add_subplot(rows, 2, i)
        plt.axis("off")
        plt.imshow(
            torch.permute(transforms.ToTensor()(skcolor.lab2rgb(torch.permute(full_image, (1, 2, 0)))), (1, 2, 0)))
        figure.add_subplot(rows, 2, i + 1)
        plt.axis("off")
        plt.imshow(bw, cmap='gray')
    plt.show()


if __name__ == '__main__':
    print("Loading")
    train_loader, test_loader, train_len, test_len = get_datasets()
    print("Loaded")
    for i,x in enumerate(train_loader):
        print(i)
