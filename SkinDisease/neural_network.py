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
        self.fc1 = nn.Linear(16 * 125 * 125, 2048)  # Input features count assumes images rezied to 512 x 512
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x)) # RELU activation functions for all layers
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x