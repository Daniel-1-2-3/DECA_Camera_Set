import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # Extract spatial features, high level
        self.pool = nn.MaxPool2d(2, 2) # Downsamples image, reduces spatial complexity
        self.conv2 = nn.Conv2d(6, 16, 5) # Extract spatial features, lower level
        # Fully connected layers, 5 to detect subtle patterns in skin dataset
        self.fc1 = nn.Linear(32 * 75 * 75, 1024)  
        self.fc2 = nn.Linear(1024, 512)  
        self.fc3 = nn.Linear(512, 256)  
        self.fc4 = nn.Linear(256, 14) # Output layer for 14 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x)) # RELU activation functions for all layers
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x