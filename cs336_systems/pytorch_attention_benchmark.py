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
from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.optimizer import AdamW

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

def benchmark_transformer(
    model_size: str,
    seq_len: int,
    batch_size: int = 4,
    num_steps: int = 100,
    warmup_steps: int = 10,
    compiled: bool = False
) -> Tuple[float, float, float]:
    """
    Benchmark full Transformer model.
    
    Args:
        model_size: Size of model to benchmark
        seq_len: Sequence length
        batch_size: Batch size
        num_steps: Number of steps to measure
        warmup_steps: Number of warmup steps
        compiled: Whether to use compiled version
        
    Returns:
        Tuple of (forward time per step, backward time per step, total time per step)
    """
    # Model configurations
    MODEL_CONFIGS = {
        "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
        "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
        "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
        "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
        "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    }
    
    # Initialize model
    model = Transformer(
        vocab_size=10000,
        context_length=seq_len,
        rope_theta=10000.0,
        **MODEL_CONFIGS[model_size]
    ).cuda()
    
    if compiled:
        model = torch.compile(model)
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    # Create random batch
    x = torch.randint(0, 10000, (batch_size, seq_len), device='cuda', dtype=torch.long)
    y = torch.randint(0, 10000, (batch_size, seq_len), device='cuda', dtype=torch.long)
    
    # Warmup
    for _ in range(warmup_steps):
        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        torch.cuda.synchronize()
    
    # Time forward passes
    forward_times = []
    for _ in range(num_steps):
        start = timeit.default_timer()
        logits = model(x)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        forward_times.append(end - start)
    
    # Time backward passes
    backward_times = []
    total_times = []
    for _ in range(num_steps):
        # Forward pass
        start = timeit.default_timer()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        
        # Calculate times
        total_times.append(end - start)
        backward_times.append(end - start - forward_times[-1])
    
    # Calculate average times
    avg_forward = sum(forward_times) / num_steps
    avg_backward = sum(backward_times) / num_steps
    avg_total = sum(total_times) / num_steps
    
    return avg_forward, avg_backward, avg_total

def main() -> None:
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark PyTorch attention and Transformer models")
    parser.add_argument("--benchmark-type", choices=["attention", "transformer"], required=True,
                       help="Type of benchmark to run")
    parser.add_argument("--model-size", type=str, choices=["small", "medium", "large", "xl", "2.7B"],
                       help="Model size for transformer benchmark")
    args = parser.parse_args()
    
    if args.benchmark_type == "attention":
        # Hidden dimensions and sequence lengths to test
        d_models = [1024, 2048, 4096, 8192]
        seq_lens = [1024, 4096, 8192, 16384]
        
        # Create results table
        results = []
        
        # Run benchmarks for both compiled and uncompiled versions
        for compiled in [False, True]:
            for d_model in d_models:
                for seq_len in seq_lens:
                    print(f"\nBenchmarking {'compiled' if compiled else 'uncompiled'} attention")
                    print(f"d_model={d_model}, seq_len={seq_len}")
                    try:
                        forward_time, backward_time, peak_memory = benchmark_attention(
                            d_model=d_model,
                            seq_len=seq_len,
                            compiled=compiled
                        )
                        
                        results.append({
                            'version': 'compiled' if compiled else 'uncompiled',
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
                                'version': 'compiled' if compiled else 'uncompiled',
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
        df = pd.DataFrame(results)
        df.to_csv('attention_benchmark_results.csv', index=False)
        print("\nResults saved to attention_benchmark_results.csv")
        
        # Print LaTeX table
        print("\nLaTeX table:")
        print(df.to_latex(index=False, float_format=lambda x: f"{x:.2f}"))
    
    else:  # transformer benchmark
        if not args.model_size:
            raise ValueError("Model size must be specified for transformer benchmark")
        
        # Sequence lengths to test
        seq_lens = [128, 256, 512]
        
        # Create results table
        results = []
        
        # Run benchmarks for both compiled and uncompiled versions
        for compiled in [False, True]:
            for seq_len in seq_lens:
                print(f"\nBenchmarking {'compiled' if compiled else 'uncompiled'} transformer")
                print(f"model_size={args.model_size}, seq_len={seq_len}")
                
                forward_time, backward_time, total_time = benchmark_transformer(
                    model_size=args.model_size,
                    seq_len=seq_len,
                    compiled=compiled
                )
                
                results.append({
                    'version': 'compiled' if compiled else 'uncompiled',
                    'model_size': args.model_size,
                    'seq_len': seq_len,
                    'forward_time_ms': forward_time * 1000,
                    'backward_time_ms': backward_time * 1000,
                    'total_time_ms': total_time * 1000
                })
                
                print(f"Forward time: {forward_time*1000:.2f}ms")
                print(f"Backward time: {backward_time*1000:.2f}ms")
                print(f"Total time: {total_time*1000:.2f}ms")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(results)
        df.to_csv('transformer_benchmark_results.csv', index=False)
        print("\nResults saved to transformer_benchmark_results.csv")
        
        # Print LaTeX table
        print("\nLaTeX table:")
        print(df.to_latex(index=False, float_format=lambda x: f"{x:.2f}"))

if __name__ == "__main__":
    main() 