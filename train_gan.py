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


def get_gen_optimizer(vgg_bottom, gen):
    params = list(vgg_bottom.parameters()) + list(gen.parameters())
    return torch.optim.Adam(params, lr=0.00002, betas=(0.5, 0.999))


def get_disc_optimizer(discriminator):
    return torch.optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.9, 0.999))


def get_gen_criterion():
    kld = torch.nn.KLDivLoss(reduction='batchmean')

    def loss_function(ab, classes, discrim, true_ab, true_labels):
        mse_loss = nn_utils.mse(ab, true_ab)
        kld_loss = kld(torch.log(classes),
                       functional.softmax(true_labels, dim=1))
        wasser_loss = nn_utils.wasserstein_loss(discrim)
        print("mse: " + str(mse_loss.item()))
        print("kld: " + str(kld_loss.item()))
        print("wasser: " + str(wasser_loss.item()))
        loss = 1 * mse_loss \
            + 0.003 * kld_loss \
            + 0.1 * wasser_loss
        return loss

    return loss_function


def get_disc_criterion(device='cpu'):
    def loss_function(real, pred, real_sample, pred_sample, discriminator,
                      gradient_penalty_weight=1):
        real_loss = nn_utils.wasserstein_loss(real)
        pred_loss = nn_utils.wasserstein_loss(pred)
        gp_loss = nn_utils.compute_gradient_penalty(
            discriminator, real_sample, pred_sample, device=device) * gradient_penalty_weight
        # gp_loss = nn_utils.gradient_penalty_loss(avg, random_average_ab, gradient_penalty_weight)

        print("real: " + str(real_loss.item()))
        print("pred: " + str(pred_loss.item()))
        print("grad: " + str(gp_loss.item()))
        return 1 * real_loss + \
            -1 * pred_loss + \
            1 * gp_loss

    return loss_function


def generate_from_bw(device, vgg_bottom, unflatten, generator, grey):
    grey_3 = grey.repeat(1, 3, 1, 1).to(device)
    vgg_bottom_out_flat = vgg_bottom(grey_3)
    # To undo the flatten operation in vgg_bottom
    vgg_bottom_out = unflatten(vgg_bottom_out_flat)
    predicted_ab, _ = generator(vgg_bottom_out)
    return predicted_ab


def train_gan(e=None, b=None):
    # Get cpu or gpu device for training.
    device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
    print(device)

    print("Loading data...")
    train_loader, test_loader, train_len, test_len = dataset.get_datasets()
    print("Loaded")
    save_models_path = os.path.join(config.model_dir)
    if not os.path.exists(save_models_path):
        os.makedirs(save_models_path)

    # Load models
    vgg_bottom, unflatten = partial_vgg.get_partial_vgg()
    vgg_bottom, unflatten = vgg_bottom.to(device), unflatten.to(device)
    # Yes it's strange that the bottom gets trained but the top doesn't
    vgg_top = partial_vgg.get_vgg_top().to(device)
    discriminator = network.Discriminator().to(device)
    generator = network.Colorization_Model().to(device)

    if e is not None and b is not None:
        vgg_bottom.load_state_dict(torch.load(
            "models/vgg_bottom_e"+str(e)+"_b"+str(b)+".pth"))
        discriminator.load_state_dict(torch.load(
            "models/discriminator_e"+str(e)+"_b"+str(b)+".pth"))
        generator.load_state_dict(torch.load(
            "models/generator_e"+str(e)+"_b"+str(b)+".pth"))
    else:
        e = 0
        b = 0
    gen_optimizer = get_gen_optimizer(vgg_bottom, generator)
    disc_optimizer = get_disc_optimizer(discriminator)
    gen_criterion = get_gen_criterion()
    disc_criterion = get_disc_criterion(device)

    num_batches = int(len(train_loader) / config.batch_size)
    # torch.autograd.set_detect_anomaly(True)
    with open('logging.txt', 'w') as log, open('errors.txt', 'w') as err:
        sys.stdout = log
        sys.stderr = err
        print("New Training Sequence:")
        for epoch in range(e, config.num_epochs):
            #print("Training epoch " + str(epoch))
            running_gen_loss = 0.0
            running_disc_loss = 0.0
            for i, data in enumerate(train_loader, 0)[b:]:
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
                gen_loss = gen_criterion(predicted_ab, predicted_classes, discrim_from_predicted,
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
    if len(sys.argv) == 2:
        epoch = args[0]
        batch = args[1]
        train_gan(epoch=epoch, batch=batch)
    else:
        train_gan()
