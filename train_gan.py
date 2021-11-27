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
    return torch.optim.Adam(discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))


def get_gen_criterion():
    kld = torch.nn.KLDivLoss(reduction='batchmean')

    def loss_function(ab, classes, discrim, true_ab, true_labels, true_discrim):
        loss = 1 * nn_utils.mse(ab, true_ab) + 0.003 * kld(torch.log(classes), functional.softmax(true_labels,
                                                                                                  dim=1)) - 0.1 * nn_utils.wasserstein_loss(
            discrim, true_discrim)
        return loss

    return loss_function


def get_disc_criterion():
    def loss_function(real, pred, avg, pos, neg, dummy, random_average_ab,
                      gradient_penalty_weight=10):
        return -1 * nn_utils.wasserstein_loss(real, pos) + \
               1 * nn_utils.wasserstein_loss(pred, neg) + \
               1 * nn_utils.gradient_penalty_loss(avg, dummy, random_average_ab, gradient_penalty_weight)

    return loss_function


def generate_from_bw(device, vgg_bottom, unflatten, generator, grey):
    grey_3 = grey.repeat(1, 3, 1, 1).to(device)
    vgg_bottom_out_flat = vgg_bottom(grey_3)
    # To undo the flatten operation in vgg_bottom
    vgg_bottom_out = unflatten(vgg_bottom_out_flat)
    predicted_ab, _ = generator(vgg_bottom_out)
    return predicted_ab


def train_gan():
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data...")
    train_loader, test_loader, train_len, test_len = dataset.get_datasets()
    print("Loaded")
    save_models_path = os.path.join(config.model_dir, config.test_name)
    if not os.path.exists(save_models_path):
        os.makedirs(save_models_path)

    # Load models
    vgg_bottom, unflatten = partial_vgg.get_partial_vgg()
    vgg_top = partial_vgg.get_vgg_top()  # Yes it's strange that the bottom gets trained but the top doesn't
    discriminator = network.Discriminator()
    generator = network.Colorization_Model()

    # Real, Fake and Dummy for Discriminator
    positive_y = np.ones((config.batch_size, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((config.batch_size, 1), dtype=np.float32)

    gen_optimizer = get_gen_optimizer(vgg_bottom, generator)
    disc_optimizer = get_disc_optimizer(discriminator)
    gen_criterion = get_gen_criterion()
    disc_criterion = get_disc_criterion()

    demo_data = next(iter(test_loader))
    demo_bw = demo_data[1][0]

    num_batches = int(len(train_loader) / config.batch_size)
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(config.num_epochs):
        print("Training epoch " + str(epoch))
        running_gen_loss = 0.0
        running_disc_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            # Print progress
            # sys.stdout.write('\r')
            # sys.stdout.write("[%-20s] %d%%" % ('='*((50*i)//num_batches), 100* i//num_batches))
            # sys.stdout.flush()
            print("Batch " + str(i))

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
            random_average_ab = nn_utils.random_weighted_average(predicted_ab, ab)

            discrim_from_real = discriminator(torch.concat([grey, ab], dim=1))
            discrim_from_predicted = discriminator(torch.concat([grey, predicted_ab], dim=1))
            discrim_from_avg = discriminator(torch.concat([grey, random_average_ab], dim=1))

            # Train generator
            gen_loss = gen_criterion(predicted_ab, predicted_classes, discrim_from_predicted,
                                     ab, vgg_out, positive_y)
            gen_optimizer.zero_grad()
            gen_loss.backward(retain_graph=True)
            running_gen_loss = running_gen_loss + gen_loss.item()

            # Train discriminator
            disc_loss = disc_criterion(discrim_from_real, discrim_from_predicted, discrim_from_avg,
                                       positive_y, negative_y, dummy_y, random_average_ab)
            disc_optimizer.zero_grad()
            disc_loss.backward()
            running_disc_loss = running_disc_loss + gen_loss.item()

            gen_optimizer.step()
            disc_optimizer.step()

            # Save a demo image after every 10 batches, for testing
            if i % 10 == 0:
                demo_ab = generate_from_bw(device, vgg_bottom, unflatten, generator, demo_bw)
                # Reshape dimensions to be as expected
                processed_ab = torch.squeeze(demo_ab, dim=0)
                processed_image = dataset.to_image(demo_bw, processed_ab)
                plt.imsave("test_output/e" + str(epoch) + "b" + str(i) + ".png", processed_image.numpy())

        torch.save(vgg_bottom.state_dict(), save_models_path + "/vgg_bottom_" + str(epoch) + ".pth")
        torch.save(generator.state_dict(), save_models_path + "/generator_" + str(epoch) + ".pth")
        torch.save(discriminator.state_dict(), save_models_path + "/discriminator_" + str(epoch) + ".pth")


if __name__ == '__main__':
    train_gan()
