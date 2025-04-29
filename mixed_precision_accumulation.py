import torch

# Test 1: All FP32
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print("FP32 accumulation:", s.item())

# Test 2: All FP16
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print("FP16 accumulation:", s.item())

# Test 3: FP32 accumulator with FP16 additions
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print("FP32 accumulator with FP16 additions:", s.item())

# Test 4: FP32 accumulator with explicit FP16->FP32 conversion
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print("FP32 accumulator with explicit conversion:", s.item()) 