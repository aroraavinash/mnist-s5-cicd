import unittest
import torch
from src.model import MNISTModel
from src.utils import count_parameters

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = MNISTModel()
        
    def test_parameter_count(self):
        param_count = count_parameters(self.model)
        self.assertLess(param_count, 100000, 
                       f"Model has {param_count} parameters, should be less than 100000")
        
    def test_input_shape(self):
        test_input = torch.randn(1, 1, 28, 28)
        try:
            output = self.model(test_input)
        except:
            self.fail("Model failed to process 28x28 input")
            
    def test_output_shape(self):
        test_input = torch.randn(1, 1, 28, 28)
        output = self.model(test_input)
        self.assertEqual(output.shape[1], 10, 
                        f"Model output should have 10 classes, got {output.shape[1]}")

if __name__ == '__main__':
    unittest.main() 