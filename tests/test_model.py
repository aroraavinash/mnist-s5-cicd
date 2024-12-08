import unittest
import torch
from torchvision import datasets, transforms
from src.model import MNISTModel
from src.utils import count_parameters
from src.train import train_epoch

class TestModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MNISTModel().to(self.device)
        
    def test_parameter_count(self):
        param_count = count_parameters(self.model)
        self.assertLess(param_count, 25000, 
                       f"Model has {param_count} parameters, should be less than 25000")
        
    def test_single_epoch_accuracy(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transform),
            batch_size=128, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        accuracy = train_epoch(self.model, self.device, train_loader, optimizer)
        
        self.assertGreaterEqual(accuracy, 95.0,
            f"Model accuracy {accuracy:.2f}% is below the required 95% in single epoch")

if __name__ == '__main__':
    unittest.main() 