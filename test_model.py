import torch
from model.model import create_model
from torchvision import datasets, transforms

def test_model():
    print("Running test: Model Parameter Count")
    
    model = create_model()
    model.train()  # Set the model to training mode

    # Define a loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Assuming a classification task
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # You can adjust the learning rate

    # Load the MNIST dataset for training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize for single channel (grayscale)
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Training loop
    for epoch in range(1):  # Number of epochs, adjust as needed
        running_loss = 0.0
        for data, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(data)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    model.eval()  # Set the model to evaluation mode

    # Test 1: Check number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    assert num_params < 100000, f"Model has {num_params} parameters, exceeds limit."

    print("Running test: Model Input Shape")
    
    # Test 2: Check input shape
    dummy_input = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output = model(dummy_input)
    assert output.shape[1] == 10, "Output shape is incorrect, expected 10 outputs."

    print("Running test: Model Accuracy")
    
    # Load the MNIST dataset for testing
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)  # Total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

    accuracy = 100 * correct / total  # Calculate accuracy
    print(f"Model accuracy: {accuracy}%")
    assert accuracy >= 80, f"Model accuracy is below 80%: {accuracy}%."

def test_model_accuracy():
    print("Running test: Model Accuracy")
    # ... existing test code ...

def test_model_loss():
    print("Running test: Model Loss")
    # ... existing test code ...

# Add more tests as needed

if __name__ == "__main__":
    test_model()
