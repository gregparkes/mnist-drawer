"""Run this script to re-make the model.

Requires the MNIST dataset in the data/ folder."""

import numpy as np
import os
import torch
from torch import nn, optim
from torch.autograd import Variable

from ._model import LeNet, ModelParams
from ._load_data import MNISTData
from ._fixed_queue import FixedQueue


def retrain_pyt(gui_elems,
                mnist: MNISTData,
                params: ModelParams = None,
                verbose: bool = False):

    pbar_gui, epoch_text, time_text, loss_text = gui_elems
    ROOT = "./data"
    MODEL = "./models"

    if not params:
        params = ModelParams(128)

    if not os.path.exists(ROOT):
        os.mkdir(ROOT)

    # check whether cuda is available
    use_cuda = torch.cuda.is_available() and params.is_gpu

    NUMBER_OF_BATCHES = 60000 // params.batch_size

    train_loader = mnist.train_load
    test_loader = mnist.test_load

    # define a model to train.
    model = LeNet(params.input_dims)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=params.momentum)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    
    ALL_STEPS = params.n_epochs * NUMBER_OF_BATCHES

    try:
        from tqdm import tqdm
        pbar = tqdm(total=ALL_STEPS, desc="CNN", position=0)
        has_tqdm = True
    except ImportError:
        print("tqdm not installed, using iterator.")
        has_tqdm = False

    # define time parameters.
    time_stamp_q = FixedQueue(100)
    total_steps = 0
    est_time_remaining = 0.

    # loop over some epochs.
    for epoch in range(params.n_epochs):
        running_loss = 0.
        # update epoch text.
        epoch_text.update(value=f"Epoch {epoch+1}/{params.n_epochs}")

        # set the model to training mode.
        model.train()
        # training over the batches. One epoch.
        for batch_idx, (x, target) in enumerate(train_loader):

            optimizer.zero_grad()

            if use_cuda:
                x, target = x.cuda(), target.cuda()
            
            x, target = Variable(x), Variable(target)
            # generate predictions
            out = model(x)
            # compute the negative log likelihood loss.
            loss = criterion(out, target)
            # perform backprop
            loss.backward()
            # update optimizer
            optimizer.step()

            if has_tqdm:
                pbar.update(1)
            
            # update GUI progress bar
            pbar_gui.update(current_count=batch_idx + 1)
            time_diff = pbar.last_print_t - pbar.start_t
            # add to the time stamp
            time_stamp_q.put(time_diff)

            # estimate the total time this might take.
            time_text.update(value="Time: {:0.2f}s / Est. time remaining: {:0.2f}s".format(
                time_diff, est_time_remaining))
            
            running_loss += loss.item()
            total_steps += 1
        
        # compute estimate of time remaining
        est_time_remaining = -np.mean(np.diff(time_stamp_q.elements)) * (ALL_STEPS - total_steps)
        time_text.update(value="Time: {:0.2f}s / Est. time remaining: {:0.2f}s".format(
            time_diff, est_time_remaining))

        # compute test accuracy on batch

        # set the model to evaluation mode.
        model.eval()
        total_acc = 0.

        with torch.no_grad():  
            for batch_idx, (x, target) in enumerate(test_loader):
                if use_cuda:
                    x, target = x.cuda(), target.cuda()
                x, target = Variable(x), Variable(target)
                # generate predictions.
                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                # accuracy as a proportion.
                total_acc += torch.sum(target == preds)
        
        accuracy = total_acc / 10000.
        
        loss_text.update(value="Loss: {:0.2f}, Accuracy: {:0.2f}%".format(
            running_loss, accuracy * 100.))

    if has_tqdm:
        pbar.close()

    if params.override_saved_model:
        # save the torch model.
        SAVE_PATH = os.path.abspath(os.path.join(MODEL, "mnist_cnn.pt"))
        print(f"Saving torch model to {SAVE_PATH}")
        torch.save(model.state_dict(), SAVE_PATH)
