import torch
import torch.nn as t_nn
import torch.nn.functional as t_functional
from utilities.constants import *


class TorchNeuralNetwork1(t_nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        kernel_size1 = 3
        kernel_size2 =3
 
        # n_layers = 4

        sum_kernels = kernel_size1 + kernel_size2

        filter1 = 8
        filter2 = 16

        # Define the components of the CNN
        self.conv1 = t_nn.Conv2d(1, filter1, kernel_size1)
        self.conv2 = t_nn.Conv2d(filter1, filter2, kernel_size2)
        
        self.pool = t_nn.MaxPool2d(2, stride=1)
        
        self.fc1 = t_nn.Linear(filter2*(IMAGE_WIDTH - sum_kernels)*
            (IMAGE_HEIGHT - sum_kernels), 84)
        self.fc2 = t_nn.Linear(84, num_classes)


    def forward(self, x):
        # Apply each layer to the input image x
        x = self.conv1(x)
        x = t_functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = t_functional.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = t_functional.relu(x)
        x = self.fc2(x)

        return x

class TorchNeuralNetwork2(t_nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        kernel_size1 = 12
        kernel_size2 = 6

        sum_kernels = kernel_size1 + kernel_size2

        filter1 = 8
        filter2 = 16

        # Define the components of the CNN
        self.conv1 = t_nn.Conv2d(1, filter1, kernel_size1)
        self.conv2 = t_nn.Conv2d(filter1, filter2, kernel_size2)
        
        self.pool = t_nn.MaxPool2d(2, stride=1)
        
        self.fc1 = t_nn.Linear(filter2*(IMAGE_WIDTH - sum_kernels)*
            (IMAGE_HEIGHT - sum_kernels), 84)
        self.fc2 = t_nn.Linear(84, num_classes)

        self.fc1_bn = t_nn.BatchNorm1d(84)
        self.dropout = t_nn.Dropout(p=0.4)

    def forward(self, x):
        # Apply each layer to the input image x
        x = self.conv1(x)
        x = t_functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = t_functional.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = t_functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class TorchNeuralNetwork3(t_nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        kernel_size1 = 12
        kernel_size2 = 6
        kernel_size3 = 3

        sum_kernels = kernel_size1 + kernel_size2 + kernel_size3

        filter1 = 8
        filter2 = 16
        filter3 = 32

        # Define the components of the CNN
        self.conv1 = t_nn.Conv2d(1, filter1, kernel_size1)
        self.conv2 = t_nn.Conv2d(filter1, filter2, kernel_size2)
        self.conv3 = t_nn.Conv2d(filter2, filter3, kernel_size3)
        
        self.pool = t_nn.MaxPool2d(2, stride=1)
        
        self.fc1 = t_nn.Linear(filter3*(IMAGE_WIDTH - sum_kernels)*
            (IMAGE_HEIGHT - sum_kernels), 120)
        self.fc2 = t_nn.Linear(120, 84)
        self.fc3 = t_nn.Linear(84, num_classes)

        self.fc1_bn = t_nn.BatchNorm1d(120)
        self.fc2_bn = t_nn.BatchNorm1d(84)
        self.dropout = t_nn.Dropout(p=0.4)

    def forward(self, x):
        # Apply each layer to the input image x
        x = self.conv1(x)
        x = t_functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = t_functional.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = t_functional.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = t_functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = t_functional.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class TorchNeuralNetwork4(t_nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        kernel_size1 = 12
        kernel_size2 = 6
        kernel_size3 = 3
        kernel_size4 = 3
        # n_layers = 4

        sum_kernels = kernel_size1 + kernel_size2 + kernel_size3 + kernel_size4

        filter1 = 8
        filter2 = 16
        filter3 = 32
        filter4 = 64

        # Define the components of the CNN
        self.conv1 = t_nn.Conv2d(1, filter1, kernel_size1)
        self.conv2 = t_nn.Conv2d(filter1, filter2, kernel_size2)
        self.conv3 = t_nn.Conv2d(filter2, filter3, kernel_size3)
        self.conv4 = t_nn.Conv2d(filter3, filter4, kernel_size4)
        
        self.pool = t_nn.MaxPool2d(2, stride=1)
        
        self.fc1 = t_nn.Linear(filter4*(IMAGE_WIDTH - sum_kernels)*
            (IMAGE_HEIGHT - sum_kernels), 120)
        self.fc2 = t_nn.Linear(120, 84)
        self.fc3 = t_nn.Linear(84, num_classes)

        self.fc1_bn = t_nn.BatchNorm1d(120)
        self.fc2_bn = t_nn.BatchNorm1d(84)
        self.dropout = t_nn.Dropout(p=0.4)

    def forward(self, x):
        # Apply each layer to the input image x
        x = self.conv1(x)
        x = t_functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = t_functional.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = t_functional.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = t_functional.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = t_functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = t_functional.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x
