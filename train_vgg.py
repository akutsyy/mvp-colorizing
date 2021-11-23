import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from datetime import datetime
from torch.utils.data import DataLoader

import partial_vgg


# Hyperparams (move to text file later)
class Configuration():
    def __init__(self):
        self.batch_size = 64
        self.epochs = 5
        self.IMAGE_SIZE = 224
        self.nThreads = 4
        self.learning_rate = 0.01


config = Configuration()


def test_model(model, testloader, device, num=None):
    correct = 0
    total = 0
    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(testloader):
            if num is not None and i > num:
                break

            images, labels = data[0].to(device), data[1].to(device)
            # Calculate outputs by running images through the network
            outputs = model(images)
            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print('Accuracy of the network on the %d test images: %d %%' % (num,
    #    100 * correct / total))
    return 100 * correct / total


def train_model(model, optimizer, training_dataset, test_dataset, criterion, device, epochs=2,
                model_name="unnamed_model", scheduler=None):
    accuracies = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(training_dataset, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data(running_loss)

        if scheduler is not None:
            scheduler.step()
        accuracy = test_model(model, test_dataset, device, 1000)
        accuracies.append(accuracy)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('%s: [%d, %s] accuracy: %.5f' % (model_name, epoch + 1, current_time, accuracy))

    print('Finished Training')
    return accuracies


def load_imagenet(batch_size, workers):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    evaltransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # Load a dataset
    imagenet_data = torchvision.datasets.ImageNet(root='dataset', split='train',
                                                  transform=transform)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=workers)

    testset = torchvision.datasets.ImageNet(root='dataset', split='val',
                                            transform=evaltransform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=workers)
    return data_loader, test_loader


# Define basic training loop for testing
def train_vgg():
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    data_loader, test_loader = load_imagenet(batch_size=config.batch_size, workers=config.nThreads)

    vgg = partial_vgg.VGG16(full_model=True)
    optimizer = optim.Adam(vgg.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    critereon = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    accuracies = train_model(vgg, optimizer, data_loader, test_loader, critereon, device, epochs=config.epochs,
                             model_name='vgg', scheduler=scheduler)

    # Save the Trained Model
    torch.save(vgg.state_dict(), 'weights,vgg.pkl')
