import torch
import torch.nn as t_nn
import torch.nn.functional as t_functional
from utilities.constants import *


class TorchNeuralNetwork(t_nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 4
        n_layers = 3

        # Define the components of the CNN
        self.conv1 = t_nn.Conv2d(1, 8, kernel_size)
        self.conv2 = t_nn.Conv2d(8, 16, kernel_size)
        self.conv3 = t_nn.Conv2d(16, 32, kernel_size)

        self.pool = t_nn.MaxPool2d(2, stride=1)
        
        self.fc1 = t_nn.Linear(32*(IMAGE_WIDTH - n_layers*kernel_size)*
            (IMAGE_HEIGHT - n_layers*kernel_size), 60)
        self.fc2 = t_nn.Linear(60, NUM_CLASSES)

    def forward(self, x):
        # Apply each layer to the input image x
        x = self.pool(t_functional.relu(self.conv1(x)))
        x = self.pool(t_functional.relu(self.conv2(x)))
        x = self.pool(t_functional.relu(self.conv3(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = t_functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
