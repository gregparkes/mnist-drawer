"""Loading datasets"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def dataset_mnist_pytorch(root = "./data", normalize=True):
    if normalize:
        trans = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((.5,), (1.,))])
    else:
        trans = transforms.Compose([transforms.ToTensor()])

    mnist_train = datasets.MNIST(root=root, train=True, download=True, transform=trans)
    mnist_test = datasets.MNIST(root=root, train=False, download=True, transform=trans)

    return mnist_train, mnist_test


def loader_mnist_pytorch(root ="./data", batch_size = 32):

    mnist_train, mnist_test = dataset_mnist_pytorch(root)

    train_loader = torch.utils.data.DataLoader(
        dataset=mnist_train, batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=mnist_test, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader
