import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out
    
    def name(self):
        return "LeNet"


def kerasmodel(dim, classes):
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

    model = Sequential([
        Input((*dim, 1)),
        Conv2D(16, kernel_size=(3, 3), activation="relu", name="conv1"),
        MaxPooling2D((2, 2), name="pool1"),
        Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv2"),
        MaxPooling2D((2, 2), name="pool2"),
        Flatten(),
        Dense(64, activation="relu", name="dense1"),
        Dropout(0.2, name="drop1"),
        Dense(32, activation="relu", name="dense2"),
        Dense(classes, activation="softmax")
    ])
    return model


