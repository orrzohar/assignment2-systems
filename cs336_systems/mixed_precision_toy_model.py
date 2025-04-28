import torch
import torch.nn as nn
import torch.cuda.amp as amp

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

def print_tensor_info(name, tensor):
    print(f"{name}: dtype={tensor.dtype}, shape={tensor.shape}")

def test_mixed_precision():
    # Create model and move to GPU
    model = ToyModel(5, 3).cuda()
    
    # Create input and target
    x = torch.randn(2, 5).cuda()
    target = torch.randint(0, 3, (2,)).cuda()
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Enable mixed precision
    with amp.autocast(device_type='cuda', dtype=torch.float16):
        # Forward pass
        print("\nDuring forward pass:")
        print_tensor_info("Model parameters (fc1.weight)", model.fc1.weight)
        
        x1 = model.fc1(x)
        print_tensor_info("Output of fc1", x1)
        
        x2 = model.ln(x1)
        print_tensor_info("Output of layer norm", x2)
        
        logits = model.fc2(x2)
        print_tensor_info("Model logits", logits)
        
        loss = criterion(logits, target)
        print_tensor_info("Loss", loss)
        
        # Backward pass
        loss.backward()
        print("\nAfter backward pass:")
        print_tensor_info("Gradient of fc1.weight", model.fc1.weight.grad)
        print_tensor_info("Gradient of fc2.weight", model.fc2.weight.grad)

if __name__ == "__main__":
    test_mixed_precision() 