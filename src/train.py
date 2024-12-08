import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTModel
from utils import count_parameters, save_model
import torch.nn.functional as F

def train_epoch(model, device, train_loader, optimizer):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
    
    return 100. * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    
    model = MNISTModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    accuracy = train_epoch(model, device, train_loader, optimizer)
    save_model(model, accuracy)
    print(f'Training Accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == '__main__':
    main() 