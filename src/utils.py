import torch
from datetime import datetime

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, accuracy, path="models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mnist_model_acc{accuracy:.2f}_{timestamp}.pth"
    torch.save(model.state_dict(), f"{path}/{filename}")
    return filename

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model 