
import time
import PySimpleGUI as sg
import numpy as np
import os
import argparse
import gzip
# pytorch
import torch

from ._load_data import dataset_mnist_pytorch, loader_mnist_pytorch
from ._model import LeNet


class RetrainWindow:

    def __init__(self):
        """Defines and initialises a modal window."""

        if os.path.isfile("models/mnist_cnn.pt"):
            DEFAULT_MODEL = os.path.abspath("models/mnist_cnn.pt")
        else:
            DEFAULT_MODEL = ""

        layout = [
            [sg.Text("CNN Retraining"), sg.In(enable_events=True, key="-OUTPUT_FILE-", default_text=DEFAULT_MODEL), sg.FileBrowse()]]

        self.window = sg.Window("MNIST CNN Retrainer", layout, modal=True, finalize=True)
    
    def mainloop(self):
        while True:
            event, values = self.window.read()
            if event in ("Exit", sg.WIN_CLOSED):
                break
        self.window.close()