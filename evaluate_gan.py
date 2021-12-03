import os
import sys

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

import benchmark_utils
import config
import dataset
import network
import partial_vgg


def demo_model_images(e,b,num=10):
    vgg_bottom, unflatten, generator, device = get_models(e,b)
    print("Loading data...")
    _,test_set = dataset.get_datasets()
    print("Loaded")

    for i in range(num):
        sample_idx = torch.randint(len(test_set), size=(1,)).item()
        real_ab, grey = test_set[sample_idx]
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

        overlay = dataset.to_image(
            torch.ones((1,224,224))*0.75, processed_ab)
        plt.imsave("demo_output/image_" + str(i) + "_overlay.png", overlay.numpy())

def re_color_video(video_tensor,vgg_bottom,unflatten,generator,device):

    video = torch.ones(video_tensor.shape[0],224,224,3)
    fcount = video.shape[0]
    for i in range(fcount//config.batch_size+(1 if fcount % config.batch_size > 0 else 0)):
        frames = video_tensor[i*config.batch_size:(i+1)*config.batch_size]
        grey = frames[:,0].unsqueeze(1).to(device)
        grey_3 = grey.repeat(1, 3, 1, 1).to(device)
        vgg_bottom_out_flat = vgg_bottom(grey_3)
        # To undo the flatten operation in vgg_bottom
        vgg_bottom_out = unflatten(vgg_bottom_out_flat)
        predicted_ab, _ = generator(vgg_bottom_out)

        processed_image = dataset.batch_to_image(grey, predicted_ab)

        video[i*config.batch_size:(i+1)*config.batch_size] = processed_image*255
    return video

def demo_model_videos(e,b,num=10,dir = "dataset/UCF101Videos_eval",outdir='demo_output'):
    vgg_bottom, unflatten, generator, device = get_models(e,b)
    filenames = os.listdir(dir)
    for i in range(num):
        sample_idx = torch.randint(len(filenames), size=(1,)).item()
        print(filenames[sample_idx])
        path = os.path.join(dir,filenames[sample_idx])
        video_tensor,audio, metadata = dataset.get_video(path)
        recolored = re_color_video(video_tensor, vgg_bottom, unflatten, generator, device)
        outpath = os.path.join(outdir,filenames[sample_idx])
        if 'audio_fps' not in metadata.keys():
            torchvision.io.write_video(filename=outpath,video_array=recolored,
                                       fps=metadata['video_fps'],video_codec='h264')
        else:
            torchvision.io.write_video(filename=outpath, video_array=recolored,
                                       fps=metadata['video_fps'], video_codec='h264',
                                       audio_array=audio,audio_fps=metadata['audio_fps'], audio_codec="mp3")


def benchmark(e,b,dir = "dataset/UCF101Videos_eval"):
    vgg_bottom, unflatten, generator, device = get_models(e,b)
    avg_snr, avg_pcpv = 0,0
    files = os.listdir(dir)
    for i,file in enumerate(files):
        print(file)
        path = os.path.join(dir,file)
        video_tensor,_, metadata = dataset.get_video(path)
        recolored = re_color_video(video_tensor, vgg_bottom, unflatten, generator, device)/255.
        recolored = torch.permute(recolored,(0,3,1,2)) # Put back into TxCxWxH

        print(torch.max(recolored[:, 0]))
        print(torch.max(recolored[:, 1]))
        print(torch.max(recolored[:, 2]))
        mse_k = torch.mean((recolored-video_tensor)**2,dim=(0,2,3)).detach().numpy()
        print(mse_k)
        max_per_channel = np.array([1.,1.,1.])
        psnr_k = 10*np.log(max_per_channel**2/mse_k)/np.log(10) # channel-wise peak signal to noise ratio
        APSNR = (psnr_k[1]+psnr_k[2])/2

        video_tensor = video_tensor.detach().numpy()
        recolored = recolored.detach().numpy()

        mean_real = np.mean(video_tensor,axis=0)
        mean_pred = np.mean(recolored,axis=0)

        sd_real = np.sum((video_tensor-mean_real)**2,axis=0)/(video_tensor.shape[0]-1)
        sd_pred = np.sum((recolored-mean_pred)**2,axis=0)/(recolored.shape[0]-1)

        sd_real_a, sd_real_b = sd_real[1], sd_real[2]
        sd_pred_a, sd_pred_b = sd_pred[1], sd_pred[2]

        sd_real_a, sd_real_b = sd_real_a.flatten(), sd_real_b.flatten()
        sd_pred_a, sd_pred_b = sd_pred_a.flatten(), sd_pred_b.flatten()

        sd_a = np.stack([sd_real_a,sd_pred_a])
        sd_b = np.stack([sd_real_b,sd_pred_b])

        corr_a = np.corrcoef(sd_a)
        corr_b = np.corrcoef(sd_b)

        PCPV = (corr_a[0][1]+corr_b[0][1])/2

        avg_snr += APSNR
        avg_pcpv += PCPV
        print(avg_snr/(i+1))
        print(avg_pcpv/(i+1))
    avg_snr = avg_snr/len(files)
    avg_pcpv = avg_pcpv/len(files)
    return avg_snr, avg_pcpv




def get_models(e, b):
    # Get cpu or gpu device for training.
    device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
    #device = "cpu"
    print(device)

    # Load models
    vgg_bottom, unflatten = partial_vgg.get_partial_vgg()
    vgg_bottom, unflatten = vgg_bottom.to(device), unflatten.to(device)
    generator = network.Colorization_Model().to(device)

    vgg_bottom.load_state_dict(torch.load(
        "/home/jlf60/mvp-colorizing/models/vgg_bottom_e"+str(e)+"_b"+str(b)+".pth",map_location=device))
    generator.load_state_dict(torch.load(
        "/home/jlf60/mvp-colorizing/models/generator_e"+str(e)+"_b"+str(b)+".pth",map_location=device))

    return vgg_bottom,unflatten,generator,device


if __name__ == '__main__':
    epoch = 1
    batch = 4199
    print(epoch)
    print(batch)
    #spacial, temporal = benchmark(e=epoch, b=batch,dir="/home/jlf60/mvp-colorizing/dataset/UCF101Videos_eval")
    demo_model_videos(epoch,batch, num=648,dir = "/home/jlf60/mvp-colorizing/dataset/UCF101Videos_eval",outdir='/home/jlf60/mvp-colorizing/demo_output')
    #args = sys.argv[1:]
    #if len(args) == 2:
        #epoch = int(args[0])
        #batch = int(args[1])
        #print(epoch)
        #print(batch)
        #spacial, temporal = benchmark(e=epoch, b=batch,dir="/home/jlf60/mvp-colorizing/dataset/UCF101Videos_eval")
        ##spacial, temporal = benchmark(e=epoch, b=batch,dir="dataset/UCF101Videos_eval")
        #print(spacial)
        #print(temporal)
        ##demo_model_videos(epoch,batch,num=10,dir = "dataset/UCF101Videos_eval",outdir='demo_output')
        
    #else:
        #print("Requires arguments: <epoch> <batch>")
