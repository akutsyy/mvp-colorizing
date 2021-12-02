import os
import sys

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from skimage import color as skcolor

import benchmark_utils
import config
import dataset
import network
import partial_vgg


def demo_model_images(e, b, num=10,trainset=False):
    vgg_bottom, unflatten, generator, device = get_models(e, b)
    print("Loading data...")
    train_set, test_set = dataset.get_datasets()
    if trainset:
        set = train_set
    else:
        set = test_set
    print("Loaded")

    for i in range(num):
        sample_idx = torch.randint(len(set), size=(1,)).item()
        data = set[sample_idx]
        real_ab, grey = data
        grey = grey.unsqueeze(0).to(device)

        grey_3 = grey.repeat(1, 3, 1, 1).to(device)
        vgg_bottom_out_flat = vgg_bottom(grey_3)
        # To undo the flatten operation in vgg_bottom
        vgg_bottom_out = unflatten(vgg_bottom_out_flat)
        predicted_ab, _ = generator(vgg_bottom_out)

        processed_ab = torch.squeeze(predicted_ab[0].detach(), dim=0)
        processed_image = dataset.to_image(
            data[1], processed_ab)
        plt.imsave("demo_output/image_" + str(i) + ".png", processed_image.numpy())

        overlay = dataset.to_image(
            torch.ones((1, 224, 224)) * 0.75, processed_ab)
        #plt.imsave("demo_output/image_" + str(i) + "_overlay.png", overlay.numpy())


def re_color_video(video_tensor, vgg_bottom, unflatten, generator, device,with_bw=False):
    video = torch.ones(video_tensor.shape)
    if with_bw:
        bw = torch.ones(video_tensor.shape)
    fcount = video.shape[0]
    for i in range(fcount // config.batch_size + (1 if fcount % config.batch_size > 0 else 0)):
        frames = video_tensor[i * config.batch_size:(i + 1) * config.batch_size]
        grey = frames[:, 0].unsqueeze(1).to(device)
        grey_3 = grey.repeat(1, 3, 1, 1).to(device)
        vgg_bottom_out_flat = vgg_bottom(grey_3)
        # To undo the flatten operation in vgg_bottom
        vgg_bottom_out = unflatten(vgg_bottom_out_flat)
        predicted_ab, _ = generator(vgg_bottom_out)

        processed_image = torch.cat([grey, predicted_ab], dim=1)

        video[i * config.batch_size:(i + 1) * config.batch_size] = processed_image
        if with_bw:
            bw[i * config.batch_size:(i + 1) * config.batch_size] = torch.cat([grey,torch.ones(predicted_ab.shape)*0.5],dim=1)
    if with_bw:
        video = torch.cat([video,bw],dim=3)
    return video


def demo_model_videos(e, b, num=10, dir="dataset/UCF101Videos_eval", outdir='demo_output',all_videos=False):
    vgg_bottom, unflatten, generator, device = get_models(e, b)
    filenames = os.listdir(dir)
    if all_videos:
        num = len(filenames)
    for i in range(num):
        if all_videos:
            sample_idx = i
        else:
            sample_idx = torch.randint(len(filenames), size=(1,)).item()
        print(filenames[sample_idx])
        path = os.path.join(dir, filenames[sample_idx])
        video_tensor, audio, metadata = dataset.get_video(path)
        recolored = re_color_video(video_tensor, vgg_bottom, unflatten, generator, device,with_bw=True)
        recolored = dataset.lab_video_to_rgb(recolored)
        outpath = os.path.join(outdir, filenames[sample_idx])
        #if 'audio_fps' in metadata.keys() and audio is not None:
        #    print(audio)
        #    torchvision.io.write_video(filename=outpath, video_array=recolored,
        #                               fps=metadata['video_fps'], video_codec='libx264',
        #                               audio_array=audio, audio_fps=metadata['audio_fps'],audio_codec='mp3')
        #else:
        torchvision.io.write_video(filename=outpath, video_array=recolored,
                                   fps=metadata['video_fps'], video_codec='libx264')


def benchmark(e, b, dir="dataset/UCF101Videos_eval"):
    vgg_bottom, unflatten, generator, device = get_models(e, b)
    avg_snr, avg_pcpv = 0, 0
    files = os.listdir(dir)
    for i, file in enumerate(files):
        print(file)
        path = os.path.join(dir, file)
        video_tensor, _, metadata = dataset.get_video(path)

        recolored = re_color_video(video_tensor, vgg_bottom, unflatten, generator, device) / 255.

        mse_k = torch.mean((recolored - video_tensor) ** 2, dim=(0, 2, 3)).detach().numpy()
        print(mse_k)
        max_per_channel = np.array([1., 1.])
        psnr_k = 10 * np.log(max_per_channel ** 2 / mse_k[1:]) / np.log(10)  # channel-wise peak signal to noise ratio
        APSNR = np.mean(psnr_k)

        video_tensor = video_tensor.detach().numpy()
        recolored = recolored.detach().numpy()

        mean_real = np.mean(video_tensor, axis=0)
        mean_pred = np.mean(recolored, axis=0)

        sd_real = np.sum((video_tensor - mean_real) ** 2, axis=0) / (video_tensor.shape[0] - 1)
        sd_pred = np.sum((recolored - mean_pred) ** 2, axis=0) / (recolored.shape[0] - 1)

        sd_real_a, sd_real_b = sd_real[1], sd_real[2]
        sd_pred_a, sd_pred_b = sd_pred[1], sd_pred[2]

        sd_real_a, sd_real_b = sd_real_a.flatten(), sd_real_b.flatten()
        sd_pred_a, sd_pred_b = sd_pred_a.flatten(), sd_pred_b.flatten()

        sd_a = np.stack([sd_real_a, sd_pred_a])
        sd_b = np.stack([sd_real_b, sd_pred_b])

        corr_a = np.corrcoef(sd_a)
        corr_b = np.corrcoef(sd_b)

        PCPV = (corr_a[0][1] + corr_b[0][1]) / 2

        avg_snr = APSNR / len(files)
        avg_pcpv = PCPV / len(files)
    return avg_snr, avg_pcpv


def get_models(e, b):
    # Get cpu or gpu device for training.
    device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
    print(device)

    # Load models
    vgg_bottom, unflatten = partial_vgg.get_partial_vgg()
    vgg_bottom, unflatten = vgg_bottom.to(device), unflatten.to(device)
    generator = network.Colorization_Model().to(device)

    vgg_bottom.load_state_dict(torch.load(
        "models/vgg_bottom_e" + str(e) + "_b" + str(b) + ".pth", map_location=device))
    generator.load_state_dict(torch.load(
        "models/generator_e" + str(e) + "_b" + str(b) + ".pth", map_location=device))

    return vgg_bottom, unflatten, generator, device


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 2:
        epoch = int(args[0])
        batch = int(args[1])
        #demo_model_images(e=epoch,b=batch,num=50,trainset=True)
        demo_model_videos(e=epoch, b=batch)
        # spacial, temporal = benchmark(e=epoch, b=batch,dir="dataset/UCF101Videos_eval_test")
        # print(spacial)
        # print(temporal)
    else:
        print("Requires arguments: <epoch> <batch>")
