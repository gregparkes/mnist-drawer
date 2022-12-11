"""Loading datasets"""

import gzip, os, numpy as np
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


def load_data_tensorflow():
    # the training data is present locally...
    if os.path.isfile("data/train-images-idx3-ubyte.gz"):
        # load directly
        # loading in training data
        with gzip.open("data/train-images-idx3-ubyte.gz","r") as f:
            image_size = 28

            f.read(16)
            buf = f.read(image_size * image_size * 60000)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(60000, image_size, image_size)
            
        with gzip.open('data/train-labels-idx1-ubyte.gz','r') as f:
            f.read(8)
            labels = np.frombuffer(f.read(60000), dtype=np.uint8)
        
        # loading in test data
        with gzip.open("data/t10k-images-idx3-ubyte.gz","r") as f:
            image_size = 28

            f.read(16)
            buf = f.read(image_size * image_size * 10000)
            test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            test_data = test_data.reshape(10000, image_size, image_size)
            
        with gzip.open('data/t10k-labels-idx1-ubyte.gz','r') as f:
            f.read(8)
            test_labels = np.frombuffer(f.read(10000), dtype=np.uint8)
    else:
        # load from the keras source
        (data,labels), (test_data,test_labels) = datasets.mnist.load_data(path="mnist.npz")
    return (data,labels), (test_data,test_labels) 
