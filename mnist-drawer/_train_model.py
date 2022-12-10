"""Run this script to re-make the model.

Requires the MNIST dataset in the data/ folder."""

import os
import numpy as np

from ._model import LeNet
from ._load_data import loader_mnist_pytorch, dataset_mnist_pytorch

def retrain_pyt(pbar_gui,
                epoch_text,
                batch_size, 
                n_epochs = 20, 
                learning_rate = 0.01, 
                momentum = 0.9,
                override_save_model = True,
                train_gpu = True):

    import torch
    import torch.optim as optim
    import torch.nn as nn
    from torch.autograd import Variable
    import torch.nn.functional as F

    ROOT = "./data"

    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
        # load dataset.
        dataset_mnist_pytorch()

    # check whether cuda is available
    use_cuda = torch.cuda.is_available() and train_gpu

    NUMBER_OF_BATCHES = 60000 // batch_size
    train_loader, test_loader = loader_mnist_pytorch(ROOT, batch_size)

    model = LeNet()
    if use_cuda:
        model = model.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_epochs * NUMBER_OF_BATCHES)
        has_tqdm = True
    except ImportError:
        print("tqdm not installed, using iterator.")
        has_tqdm = False

    model.train()

    for epoch in range(n_epochs):
        # update epoch text.
        epoch_text.update(value=f"Epoch {epoch+1}/{n_epochs}")

        # training
        ave_loss = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            
            x, target = Variable(x), Variable(target)
            out = model(x)
            loss = F.nll_loss(out, target)
            loss.backward()
            optimizer.step()

            if has_tqdm:
                pbar.update(1)
            # update GUI progress bar
            pbar_gui.update(current_count=batch_idx + 1)

    if has_tqdm:
        pbar.close()

    if override_save_model:
        # save the torch model.
        SAVE_PATH = os.path.abspath(os.path.join(ROOT, "mnist_cnn.pt"))
        print(f"Saving torch model to {SAVE_PATH}")
        torch.save(model.state_dict(), SAVE_PATH)
