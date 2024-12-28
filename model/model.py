import torch
import torch.nn as nn

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)  # Input channels = 1 for grayscale
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(14 * 14 * 20, 10*10*5)  # Adjust input size based on pooling
#         #self.fc2 = nn.Linear(64, 10)  # 10 output classes for MNIST

#     def forward(self, x):
#         x = self.pool1(nn.ReLU()(self.conv1(x)))
#         x = self.pool2(nn.ReLU()(self.conv2(x)))
#         x = self.flatten(x)
#         x = nn.ReLU()(self.fc1(x))
#         return self.fc2(x)

# def create_model():
#     return SimpleCNN()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)  # Input channels = 1 for grayscale
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
       # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(10 * 7 * 7, 128)  # Adjust input size based on pooling

        self.fc2 = nn.Linear(128, 10)  # 10 output classes for MNIST
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 10*7*7)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def create_model():
    return SimpleCNN()