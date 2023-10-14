import os
import numpy as np
from dataclasses import dataclass
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision.transforms as T


class LeNet(nn.Module):
    def __init__(self, input_dim: tuple[int, int], n_classes: int = 10):
        super().__init__()
        c1 = 32
        c2 = 64
        input_dim = np.asarray(input_dim)
        d1 = c2 * np.prod((input_dim - 4) // 2)
        d2 = 128

        self.net = nn.Sequential(
            nn.Conv2d(1, c1, 3, 1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(d2, n_classes)
        )
    
    def forward(self, x: Tensor):
        return self.net(x)
    
    def name(self) -> str:
        return "LeNet"


@dataclass
class ModelParams:
    """Model parameters to training a model."""
    batch_size: int
    n_epochs: int = 20
    learning_rate: float = 0.01
    momentum: float = 0.9
    override_saved_model: bool = True
    is_gpu: bool = True
    root: str = "./data"
    base_input_size: int = 28
    base_input_dims: tuple[int, int] = (28, 28)
    input_size: int = 24
    input_dims: tuple[int, int] = (24, 24)
    n_targets: int = 10
    tf_train = T.Compose([T.CenterCrop(24), T.RandomAffine(degrees=(-30, 30), translate=(.1, .3), scale=(.5, 1.5)), T.Resize((24, 24)), T.ToTensor(), T.Normalize((.5,), (1.,))])
    tf_test = T.Compose([T.CenterCrop(24), T.ToTensor(), T.Normalize((.5,), (1.,))])


class Model:
    """Model class to handle predictions."""
    def __init__(self, model_params: ModelParams):
        """Holds the network and parameters, and perform inferences."""
        self.params = model_params
        self.net = None
        self.is_loaded = False
        #self.device = torch.device("cuda") if torch.cuda.is_available() and model_params.is_gpu else torch.device("cpu")
        self.device = torch.device("cpu")
        self.transform = T.Compose([T.CenterCrop(model_params.input_size), T.Normalize((.5,), (1.,))])

    def load(self):
        """Called by thread.start()"""
        default_path = "models/mnist_cnn.pt"
        if os.path.isfile(default_path):
            self.net = LeNet(self.params.input_dims, self.params.n_targets).to(self.device)
            model_state = torch.load(default_path, map_location=self.device)
            self.net.load_state_dict(model_state)
            self.net.eval()
    
    def forward(self, x):
        """Forward prediction."""
        input_shape = (1, 1, self.params.base_input_size, self.params.base_input_size)
        if self.is_loaded:

            self.net.eval()
            # process x grid into torch format.
            x = torch.reshape(torch.rot90(torch.tensor(x, dtype=torch.float32)), input_shape)
            # preprocess using testing transform
            x = self.transform(x).to(self.device)

            with torch.no_grad():
                y = self.net(x)
                probs = torch.softmax(torch.flatten(y), 0).cpu().detach().numpy()
                return probs

        else:
            print("Model not loaded")
