#!/usr/bin/env python3
"""
Benchmark PyTorch attention implementation

This script benchmarks the memory usage and performance of PyTorch's attention
implementation across different configurations of hidden dimension and sequence length,
comparing both compiled and uncompiled versions.
"""
import argparse
import timeit
from typing import Tuple
import torch
import torch.nn.functional as F
import pandas as pd
from cs336_systems.model_annotated import PyTorchAttention

def benchmark_attention(
    d_model: int,
    seq_len: int,
    batch_size: int = 8,
    num_steps: int = 100,
    warmup_steps: int = 10,
    compiled: bool = False
) -> Tuple[float, float, float]:
    """
    Benchmark attention implementation.
    
    Args:
        d_model: Hidden dimension
        seq_len: Sequence length
        batch_size: Batch size
        num_steps: Number of steps to measure
        warmup_steps: Number of warmup steps
        compiled: Whether to use compiled version
        
    Returns:
        Tuple of (forward time per step, backward time per step, peak memory before backward)
    """
    # Create attention module
    attention = PyTorchAttention(d_model, compiled=compiled).cuda()
    
    # Create random inputs (batch_size, seq_len, d_model)
    Q = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
    
    # Warmup
    for _ in range(warmup_steps):
        # Forward pass
        output = attention(Q, K, V)
        
        # Backward pass
        output.sum().backward()
        Q.grad = None
        K.grad = None
        V.grad = None
        
        torch.cuda.synchronize()
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    
    # Time forward passes
    forward_times = []
    for _ in range(num_steps):
        start = timeit.default_timer()
        output = attention(Q, K, V)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        forward_times.append(end - start)
    
    # Get memory usage before backward
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
    
    # Time backward passes
    backward_times = []
    for _ in range(num_steps):
        # Create new output for each backward pass
        output = attention(Q, K, V)
        
        start = timeit.default_timer()
        output.sum().backward()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        backward_times.append(end - start)
        
        # Reset gradients
        Q.grad = None
        K.grad = None
        V.grad = None
    
    # Calculate average times
    avg_forward = sum(forward_times) / num_steps
    avg_backward = sum(backward_times) / num_steps
    
    return avg_forward, avg_backward, peak_memory

def main() -> None:
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark PyTorch attention implementation")
    parser.add_argument("--compiled", action="store_true",
                       help="Whether to use compiled version")
    args = parser.parse_args()
    
    # Hidden dimensions and sequence lengths to test (as per requirements)
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    
    # Create results table
    results = []
    
    # Run benchmark for specified version (compiled or uncompiled)
    for d_model in d_models:
        for seq_len in seq_lens:
            print(f"\nBenchmarking {'compiled' if args.compiled else 'uncompiled'} attention")
            print(f"d_model={d_model}, seq_len={seq_len}")
            try:
                forward_time, backward_time, peak_memory = benchmark_attention(
                    d_model=d_model,
                    seq_len=seq_len,
                    batch_size=8,  # Fixed batch size as per requirements
                    compiled=args.compiled
                )
                
                results.append({
                    'version': 'compiled' if args.compiled else 'uncompiled',
                    'd_model': d_model,
                    'seq_len': seq_len,
                    'forward_time_ms': forward_time * 1000,
                    'backward_time_ms': backward_time * 1000,
                    'peak_memory_MB': peak_memory,
                    'status': 'success'
                })
                
                print(f"Forward time: {forward_time*1000:.2f}ms")
                print(f"Backward time: {backward_time*1000:.2f}ms")
                print(f"Peak memory: {peak_memory:.2f}MB")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results.append({
                        'version': 'compiled' if args.compiled else 'uncompiled',
                        'd_model': d_model,
                        'seq_len': seq_len,
                        'forward_time_ms': float('nan'),
                        'backward_time_ms': float('nan'),
                        'peak_memory_MB': float('nan'),
                        'status': 'OOM'
                    })
                    print("Out of memory")
                else:
                    raise e
    
    # Convert to DataFrame and save
    output_file = f'attention_benchmark_{"compiled" if args.compiled else "uncompiled"}.csv'
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print LaTeX table
    print("\nLaTeX table:")
    print(df.to_latex(index=False, float_format=lambda x: f"{x:.2f}"))

if __name__ == "__main__":
    main() 