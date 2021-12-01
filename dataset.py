import os
import random
import time
from pathlib import Path

import cv2
import shutil
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from skimage import color as skcolor
from torch.utils.data import Dataset

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


def copy_eval_videos():
    Path("dataset/UCF101Videos_eval").mkdir(parents=True, exist_ok=True)
    filenames = set(os.listdir("dataset/UCF101Images_eval"))

    def cut_end(filename):  # in format v_ApplyEyeMakeup_g08_c02_0.png
        s = filename.split("_")
        endsize = len(s[-1]) + 1
        return filename[:-endsize]

    def add_avi(filename):
        return filename + ".avi"

    filenames = map(cut_end, filenames)
    filenames = set(map(add_avi, filenames))

    for file in filenames:
        print(file)
        shutil.copyfile("dataset/UCF101/" + file, "dataset/UCF101Videos_eval/" + file)


# Used 0.05 to generate dataset
def get_video(path):
    frames, audio, metadata = torchvision.io.read_video(path)
    video_tensor = torch.ones(len(frames), 3, 224, 224)

    for i, img in enumerate(frames):
        tensor = torch.permute(torch.Tensor(cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2LAB)), (2, 0, 1)) / 255.0
        # Random crop
        if i == 0:
            a, j, h, w = transforms.RandomCrop.get_params(tensor, output_size=(224, 224))
        image = transforms.functional.crop(tensor, a, j, h, w)
        video_tensor[i] = image
    return video_tensor, audio, metadata


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

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        image = self.other_transforms(self.to_tensor(image))

        sample = (image[1:], image[0].unsqueeze(0))

        return sample


def get_loaders():
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


def get_datasets():
    train_set = UCF101ImageDataset("dataset/UCF101Images_train")
    test_set = UCF101ImageDataset("dataset/UCF101Images_eval")
    return train_set, test_set


def display_dataset_sample():
    training_data = UCF101ImageDataset("dataset/UCF101Images_train")
    rows = 6
    figure = plt.figure(figsize=(3, 6))
    figure.set_tight_layout(True)

    for i in range(1, 12, 2):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        color, bw = training_data[sample_idx]
        full_image = torch.concat([bw * 100, color * 255 - 127], dim=0)
        bw = full_image[0]
        full_color = torch.permute(
            transforms.ToTensor()(skcolor.lab2rgb(torch.permute(full_image, (1, 2, 0)).numpy())),
            (1, 2, 0))

        figure.add_subplot(rows, 2, i)
        plt.axis("off")
        plt.imshow(full_color)
        figure.add_subplot(rows, 2, i + 1)
        plt.axis("off")
        plt.imshow(bw, cmap='gray')
    plt.show()


def to_image(bw, color):
    full_image = torch.concat([bw.to('cpu') * 100, color.to('cpu') * 255 - 127], dim=0)
    full_color = torch.permute(
        transforms.ToTensor()(skcolor.lab2rgb(torch.permute(full_image, (1, 2, 0)).detach().numpy())),
        (1, 2, 0))
    return full_color


def batch_to_image(bw, color):
    full_image = torch.concat([bw.to('cpu') * 100, color.to('cpu') * 255 - 127], dim=1)
    img_tensor = torch.ones((full_image.shape[0], 224, 224, 3))
    for i in range(full_image.shape[0]):
        img_tensor[i] = torch.permute(
            transforms.ToTensor()(skcolor.lab2rgb(torch.permute(full_image[i], (1, 2, 0)).detach().numpy())),
            (1, 2, 0))
    return img_tensor


if __name__ == '__main__':
    display_dataset_sample()
    print("Loading")
    train_loader, test_loader, train_len, test_len = get_loaders()
    print("Loaded")
    start = time.time()
    for i, x in enumerate(train_loader):
        print(str(i) + ",   " + str((time.time() - start) / (i + 1)) + " seconds per batch")
        color, bw = x
