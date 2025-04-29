import torch
import torch.nn as nn
import torch.amp as amp
import argparse
import timeit
from contextlib import nullcontext

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

def benchmark_mixed_precision(precision: str, num_steps: int = 100, warmup_steps: int = 10):
    # Create model and move to GPU
    model = ToyModel(5, 3).cuda()
    
    # Create input and target
    x = torch.randn(2, 5).cuda()
    target = torch.randint(0, 3, (2,)).cuda()
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set up autocast context
    if precision == "fp32":
        autocast_context = nullcontext()
    else:
        dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
        autocast_context = torch.amp.autocast(device_type='cuda', dtype=dtype)
    
    # Print tensor info for first pass
    print(f"\nDuring forward pass (using {precision}):")
    with autocast_context:
        print_tensor_info("Model parameters (fc1.weight)", model.fc1.weight)
        x1 = model.fc1(x)
        print_tensor_info("Output of fc1", x1)
        x2 = model.ln(x1)
        print_tensor_info("Output of layer norm", x2)
        logits = model.fc2(x2)
        print_tensor_info("Model logits", logits)
        loss = criterion(logits, target)
        print_tensor_info("Loss", loss)
    
    # Warmup
    for _ in range(warmup_steps):
        with autocast_context:
            logits = model(x)
            loss = criterion(logits, target)
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()
    
    # Time forward passes
    forward_times = []
    for _ in range(num_steps):
        start = timeit.default_timer()
        with autocast_context:
            logits = model(x)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        forward_times.append(end - start)
    
    # Time backward passes
    backward_times = []
    for _ in range(num_steps):
        start = timeit.default_timer()
        with autocast_context:
            logits = model(x)
            loss = criterion(logits, target)
            loss.backward()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        backward_times.append(end - start)
        model.zero_grad()
    
    # Calculate average times
    avg_forward = sum(forward_times) / num_steps
    avg_backward = sum(backward_times) / num_steps
    
    print(f"\nBenchmark results (using {precision}):")
    print(f"Average forward time: {avg_forward*1000:.3f}ms")
    print(f"Average backward time: {avg_backward*1000:.3f}ms")
    
    # Print gradient info after final backward pass
    print("\nAfter backward pass:")
    print_tensor_info("Gradient of fc1.weight", model.fc1.weight.grad)
    print_tensor_info("Gradient of fc2.weight", model.fc2.weight.grad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test mixed precision training')
    parser.add_argument('--precision', type=str, choices=['fp32', 'float16', 'bfloat16'], 
                      default='float16', help='Precision to use for mixed precision training')
    args = parser.parse_args()
    
    benchmark_mixed_precision(args.precision) 