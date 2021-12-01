import os

import numpy as np
import torch
import torch.nn.functional as functional
import sys

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
            grey, processed_ab)
        plt.imsave("demo_output/img_" + str(i) + ".png", processed_image.numpy())


def get_models(e, b):
    # Get cpu or gpu device for training.
    device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
    print(device)

    # Load models
    vgg_bottom, unflatten = partial_vgg.get_partial_vgg()
    vgg_bottom, unflatten = vgg_bottom.to(device), unflatten.to(device)
    generator = network.Colorization_Model().to(device)

    vgg_bottom.load_state_dict(torch.load(
        "models/vgg_bottom_e"+str(e)+"_b"+str(b)+".pth"))
    generator.load_state_dict(torch.load(
        "models/generator_e"+str(e)+"_b"+str(b)+".pth"))

    return vgg_bottom,unflatten,generator,device

    num_batches = int(len(train_loader) / config.batch_size)
    # torch.autograd.set_detect_anomaly(True)
    with open('logging.txt', 'w') as log, open('errors.txt', 'w') as err:
        sys.stdout = log
        #sys.stderr = err
        print("New Training Sequence:")
        for epoch in range(e, config.num_epochs):
            #print("Training epoch " + str(epoch))
            running_gen_loss = 0.0
            running_disc_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                if i<b:
                    continue
                # Print progress
                log.flush()
                print("Epoch "+str(epoch)+" Batch " + str(i))

                # ab channels of l*a*b color space - is color
                ab, grey = data
                ab, grey = ab.to(device), grey.to(device)
                # Images are in l*a*b* space, normalized

                # Use pre-trained VGG as in original paper
                grey_3 = grey.repeat(1, 3, 1, 1).to(device)
                vgg_bottom_out_flat = vgg_bottom(grey_3)
                # To undo the flatten operation in vgg_bottom
                vgg_bottom_out = unflatten(vgg_bottom_out_flat)
                vgg_out = vgg_top(vgg_bottom_out)
                predicted_ab, predicted_classes = generator(vgg_bottom_out)
                random_average_ab = nn_utils.random_weighted_average(
                    predicted_ab, ab, device=device)

                discrim_from_real = discriminator(
                    torch.concat([grey, ab], dim=1))
                discrim_from_predicted = discriminator(
                    torch.concat([grey, predicted_ab], dim=1))

                # Train generator
                gen_loss = gen_criterion(predicted_ab, predicted_classes,discrim_from_real, discrim_from_predicted,
                                         ab, vgg_out)
                gen_optimizer.zero_grad()
                gen_loss.backward(retain_graph=True)
                running_gen_loss = running_gen_loss + gen_loss.detach().item()

                # Train discriminator
                disc_loss = disc_criterion(discrim_from_real, discrim_from_predicted, torch.concat([grey, ab], dim=1),
                                           torch.concat([grey, predicted_ab], dim=1), discriminator)
                disc_optimizer.zero_grad()
                disc_loss.backward()
                running_disc_loss = running_disc_loss + disc_loss.detach().item()

                gen_optimizer.step()
                disc_optimizer.step()
                print("Generator loss: "+str(gen_loss.item()))
                print("Discriminator loss: "+str(disc_loss.item()))

                # Save a demo image after every 50 batches
                if i % 50 == 0:
                    # Reshape dimensions to be as expected
                    processed_ab = torch.squeeze(predicted_ab[0], dim=0)
                    processed_image = dataset.to_image(
                        data[1][0], processed_ab)
                    plt.imsave("test_output/e" + str(epoch) + "b" +
                               str(i) + ".png", processed_image.numpy())

                # Save the models every 200 batches
                if i % 200 == 199:
                    torch.save(vgg_bottom.state_dict(),
                               save_models_path + "/vgg_bottom_e" + str(epoch) + "_b" + str(i) + ".pth")
                    torch.save(generator.state_dict(),
                               save_models_path + "/generator_e" + str(epoch) + "_b" + str(i) + ".pth")
                    torch.save(discriminator.state_dict(),
                               save_models_path + "/discriminator_e" + str(epoch) + "_b" + str(i) + ".pth")

            b = 0


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 2:
        epoch = int(args[0])
        batch = int(args[1])
        eval_gan(e=epoch, b=batch)
    else:
        print("Requires arguments: <epoch> <batch>")
