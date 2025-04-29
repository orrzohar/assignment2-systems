import torch

# Test 1: FP32 accumulation
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print("FP32 accumulation:", s)

# Test 2: FP16 accumulation
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print("FP16 accumulation:", s)

# Test 3: FP32 accumulator with FP16 values
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print("FP32 accumulator with FP16 values:", s)

# Test 4: FP32 accumulator with explicit FP16->FP32 conversion
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print("FP32 accumulator with explicit FP16->FP32 conversion:", s) 