import torch
import torch.nn as t_nn
import torch.nn.functional as t_functional
from utilities.constants import *


class TorchNeuralNetwork(t_nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 4
        n_layers = 4

        filter1 = 8
        filter2 = 16
        filter3 = 32
        filter4 = 64

        # Define the components of the CNN
        self.conv1 = t_nn.Conv2d(1, filter1, kernel_size)
        self.conv2 = t_nn.Conv2d(filter1, filter2, kernel_size)
        self.conv3 = t_nn.Conv2d(filter2, filter3, kernel_size)
        self.conv4 = t_nn.Conv2d(filter3, filter4, kernel_size)
        
        self.conv1_bn = t_nn.BatchNorm2d(filter1)
        self.conv2_bn = t_nn.BatchNorm2d(filter2)
        self.conv3_bn = t_nn.BatchNorm2d(filter3)
        self.pool = t_nn.MaxPool2d(2, stride=1)
        
        self.fc1 = t_nn.Linear(filter4*(IMAGE_WIDTH - n_layers*kernel_size)*
            (IMAGE_HEIGHT - n_layers*kernel_size), 60)
        self.fc2 = t_nn.Linear(60, NUM_CLASSES)

        self.fc1_bn = t_nn.BatchNorm1d(60)

    def forward(self, x):
        # Apply each layer to the input image x
        x = self.pool(t_functional.relu(self.conv1(x)))
        x = self.pool(t_functional.relu(self.conv2(x)))
        x = self.pool(t_functional.relu(self.conv3(x)))
        x = self.pool(t_functional.relu(self.conv4(x)))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = t_functional.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return x
