import os

import numpy as np
import torch
import torch.nn.functional as functional
import sys
import av

import torchvision
from matplotlib import pyplot as plt

import config
import dataset
import network
import nn_utils
import partial_vgg
import train_gan



def demo_model_images(e,b,num=10):
    vgg_bottom, unflatten, generator, device = get_models(e,b)
    print("Loading data...")
    _,test_set = dataset.get_datasets()
    print("Loaded")

    for i in range(num):
        sample_idx = torch.randint(len(test_set), size=(1,)).item()
        _, grey = test_set[sample_idx]
        grey = grey.unsqueeze(0).to(device)

        grey_3 = grey.repeat(1, 3, 1, 1).to(device)
        vgg_bottom_out_flat = vgg_bottom(grey_3)
        # To undo the flatten operation in vgg_bottom
        vgg_bottom_out = unflatten(vgg_bottom_out_flat)
        predicted_ab, _ = generator(vgg_bottom_out)

        processed_ab = torch.squeeze(predicted_ab[0].detach(), dim=0)
        processed_image = dataset.to_image(
            torch.squeeze(grey,dim=0), processed_ab)
        plt.imsave("demo_output/image_" + str(i) + ".png", processed_image.numpy())

def re_color_video(path,vgg_bottom,unflatten,generator,device):
    video_tensor, audio, metadata = dataset.get_video(path)
    original_len = video_tensor.shape[0]

    # Pad to batch size
    video_len = config.batch_size-(video_tensor.shape[0] % config.batch_size) + video_tensor.shape[0]
    video_tensor = torch.cat([video_tensor,torch.ones((video_len-video_tensor.shape[0],3,224,224))],dim=0)
    video = torch.ones((video_len,224,224,3))
    for i in range(video_len//config.batch_size):
        frames = video_tensor[i*config.batch_size:(i+1)*config.batch_size]
        grey = frames[:,0].unsqueeze(1).to(device)
        grey_3 = grey.repeat(1, 3, 1, 1).to(device)
        vgg_bottom_out_flat = vgg_bottom(grey_3)
        # To undo the flatten operation in vgg_bottom
        vgg_bottom_out = unflatten(vgg_bottom_out_flat)
        predicted_ab, _ = generator(vgg_bottom_out)

        processed_ab = torch.squeeze(predicted_ab.detach(), dim=0)
        processed_image = dataset.batch_to_image(grey, processed_ab)

        video[i*config.batch_size:(i+1)*config.batch_size] = processed_image*255
    return video[:original_len],metadata

def demo_model_videos(e,b,num=10,dir = "dataset/UCF101Videos_eval",outdir='demo_output'):
    vgg_bottom, unflatten, generator, device = get_models(e,b)
    filenames = os.listdir(dir)
    for i in range(num):
        sample_idx = torch.randint(len(filenames), size=(1,)).item()
        print(filenames[sample_idx])
        path = os.path.join(dir,filenames[sample_idx])
        video,metadata = re_color_video(path, vgg_bottom, unflatten, generator, device)
        outpath = os.path.join(outdir,filenames[sample_idx])
        torchvision.io.write_video(filename=outpath,video_array=video,
                                   fps=metadata['video_fps'],video_codec='libx264')
def get_models(e, b):
    # Get cpu or gpu device for training.
    device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
    print(device)

    # Load models
    vgg_bottom, unflatten = partial_vgg.get_partial_vgg()
    vgg_bottom, unflatten = vgg_bottom.to(device), unflatten.to(device)
    generator = network.Colorization_Model().to(device)

    vgg_bottom.load_state_dict(torch.load(
        "models/vgg_bottom_e"+str(e)+"_b"+str(b)+".pth",map_location=device))
    generator.load_state_dict(torch.load(
        "models/generator_e"+str(e)+"_b"+str(b)+".pth",map_location=device))

    return vgg_bottom,unflatten,generator,device


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 2:
        epoch = int(args[0])
        batch = int(args[1])
        demo_model_images(e=epoch, b=batch,num=100)
    else:
        print("Requires arguments: <epoch> <batch>")
