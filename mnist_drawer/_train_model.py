"""Run this script to re-make the model.

Requires the MNIST dataset in the data/ folder."""

import numpy as np
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from ._model import LeNet
from ._load_data import loader_mnist_pytorch, dataset_mnist_pytorch
from ._fixed_queue import FixedQueue

def retrain_pyt(pbar_gui,
                epoch_text,
                time_text,
                batch_size, 
                n_epochs = 20, 
                learning_rate = 0.01, 
                momentum = 0.9,
                override_save_model = True,
                train_gpu = True):

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

    ALL_STEPS = n_epochs * NUMBER_OF_BATCHES

    try:
        from tqdm import tqdm
        pbar = tqdm(total=ALL_STEPS, desc="CNN", position=0)
        has_tqdm = True
    except ImportError:
        print("tqdm not installed, using iterator.")
        has_tqdm = False

    model.train()

    time_stamp_q = FixedQueue(100)
    total_steps = 0
    est_time_remaining = 0.

    for epoch in range(n_epochs):
        # update epoch text.
        epoch_text.update(value=f"Epoch {epoch+1}/{n_epochs}")

        # training
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
            time_diff = pbar.last_print_t - pbar.start_t
            # add to the time stamp
            time_stamp_q.put(time_diff)

            # estimate the total time this might take.
            time_text.update(value="Time: {:0.2f}s / Est. time remaining: {:0.2f}s".format(time_diff, est_time_remaining))
            total_steps += 1
        
        # compute estimate of time remaining
        est_time_remaining = -np.mean(np.diff(time_stamp_q.elements)) * (ALL_STEPS - total_steps)
        time_text.update(value="Time: {:0.2f}s / Est. time remaining: {:0.2f}s".format(time_diff, est_time_remaining))


    if has_tqdm:
        pbar.close()

    if override_save_model:
        # save the torch model.
        SAVE_PATH = os.path.abspath(os.path.join(ROOT, "mnist_cnn.pt"))
        print(f"Saving torch model to {SAVE_PATH}")
        torch.save(model.state_dict(), SAVE_PATH)
