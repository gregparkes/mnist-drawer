"""Loading datasets"""

import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader

from ._model import ModelParams


class MNISTData:
    """Object for holding the datasets and dataloader interfaces."""
    def __init__(self, model_params: ModelParams):
        self.params = model_params
        self.train_set = None
        self.test_set = None
        self.train_load = None
        self.test_load = None
        self.is_loaded = False
    
    def load(self):
        """Called using thread.start()"""
        affine = T.RandomAffine(degrees=(-30, 30), translate=(.1, .3), scale=(.5, 1.5))
        tf_train = T.Compose([T.CenterCrop(24), affine, T.Resize((24, 24)), T.ToTensor(), T.Normalize((.5,), (1.,))])
        tf_test = T.Compose([T.CenterCrop(24), T.ToTensor(), T.Normalize((.5,), (1.,))])

        self.train_set = datasets.MNIST(self.params.root, train=True, download=True,
                                        transform=tf_train)
        self.test_set = datasets.MNIST(self.params.root, train=False, download=True, transform=tf_test)

        self.train_load = DataLoader(dataset=self.train_set, batch_size=self.params.batch_size,
                                     shuffle=True, num_workers=4, 
                                     pin_memory=True, persistent_workers=True)

        self.test_load = DataLoader(dataset=self.test_set, batch_size=self.params.batch_size, 
                                    shuffle=False, pin_memory=True)
        
        self.is_loaded = True
