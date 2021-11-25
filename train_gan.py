import collections
import math

import torch
from torch import nn
import torchvision
import os
import cv2
from pathlib import Path
import random

# Hyperparams (move to text file later)
import partial_vgg


class Configuration():
    def __init__(self):
        self.batch_size = 64
        self.epochs = 5
        self.IMAGE_SIZE = 224
        self.nThreads = 4
        self.learning_rate = 0.01


config = Configuration()


def convert_videos_to_images(eval_portion):
    Path("dataset/UCF101Images_train").mkdir(parents=True, exist_ok=True)
    Path("dataset/UCF101Images_eval").mkdir(parents=True, exist_ok=True)
    total = len(os.listdir("dataset/UCF101"))
    for i, filename in enumerate(os.listdir("dataset/UCF101")):
        trimmed_filename = filename[:-4]
        print(str(100*i/total)+"%,  "+filename)
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
def load_trainset():
    convert_videos_to_images(0.05)


def train_gan():
    vgg16 = partial_vgg.get_partial_vgg()


convert_videos_to_images(0.05)