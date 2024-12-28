import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)  # Input channels = 1 for grayscale
        self.pool = nn.MaxPool2d(2, 2)
        self.pointwise1 = nn.Conv2d(10, 5, kernel_size=1)  # Reduce channels after first pooling

        self.conv2 = nn.Conv2d(5, 10, kernel_size=3, stride=1, padding=1)  # Input channels = 5 from pointwise1
        self.pointwise2 = nn.Conv2d(10, 5, kernel_size=1)  # Reduce channels after second pooling

        self.fc1 = nn.Linear(5 * 7 * 7, 128)  # Adjust input size based on pooling
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for MNIST
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pointwise1(x)  # Apply 1x1 convolution after first pooling
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pointwise2(x)  # Apply 1x1 convolution after second pooling
        x = x.view(-1, 5 * 7 * 7)  # Adjust for the new number of channels
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def create_model():
    return SimpleCNN()