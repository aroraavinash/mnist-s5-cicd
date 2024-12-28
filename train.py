import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from model.model import create_model
import datetime

def train_model():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    model = create_model()
    optimizer = optim.Adam(model.parameters())
    model.train()

    # Train model for 1 epoch
    for epoch in range(1):  # Only 1 epoch
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    # Save model with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), f'model/mnist_model_{timestamp}.pt')

if __name__ == "__main__":
    train_model()
