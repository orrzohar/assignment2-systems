#!/usr/bin/env python3
"""
Advanced Nsight Systems profiling for Transformer models

This script profiles transformer models using NVTX ranges to annotate different 
components of the execution, making it easier to analyze where time is spent
in both forward and backward passes, as well as in optimizer steps.
"""
import argparse
import timeit
from typing import Dict, Tuple, List, Optional
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.cuda.nvtx as nvtx

# Import model and replace attention function with annotated version
from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.optimizer import AdamW
from cs336_systems.model_annotated import annotated_scaled_dot_product_attention

# Replace the original attention function with the annotated version
from cs336_basics import model as basics_model
basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
print("Using annotated attention function for profiling")

# Configuration
VOCAB_SIZE = 10_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")
else:
    print(f"Warning: CUDA is not available. Using CPU instead.")
    print("Profiling with nsys requires a CUDA-capable GPU.")

# Model configurations based on Table 1 in the assignment
MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def random_batch(batch: int, seq: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random batches of token IDs for input and target."""
    x = torch.randint(0, VOCAB_SIZE, (batch, seq), device=DEVICE, dtype=torch.long)
    y = torch.randint(0, VOCAB_SIZE, (batch, seq), device=DEVICE, dtype=torch.long)
    return x, y

def run_steps(
    model: Transformer,
    x: torch.Tensor,
    y: torch.Tensor,
    n: int,
    do_backward: bool,
    run_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    profile_memory: bool = False,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Run n steps of forward/backward/optimizer and time each component.
    
    Args:
        model: The transformer model
        x: Input tokens
        y: Target tokens
        n: Number of steps to run
        do_backward: Whether to run backward pass
        run_optimizer: Whether to run optimizer step
        optimizer: Optimizer instance (required if run_optimizer is True)
        profile_memory: Whether to profile memory usage
        
    Returns:
        Tuple of (forward times, backward times, optimizer times)
    """
    fwd_times, bwd_times, optim_times = [], [], []
    criterion = torch.nn.CrossEntropyLoss()
    
    if run_optimizer and optimizer is None:
        raise ValueError("Optimizer must be provided if run_optimizer is True")
    
    for i in range(n):
        # Start step with NVTX range
        with nvtx.range(f"step_{i}"):
            # Forward pass
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            
            with nvtx.range("forward_pass"):
                start_fwd = timeit.default_timer()
                logits = model(x)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                end_fwd = timeit.default_timer()
                fwd_times.append(end_fwd - start_fwd)
            
            # Calculate loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            # Backward pass
            if do_backward:
                with nvtx.range("backward_pass"):
                    start_bwd = timeit.default_timer()
                    loss.backward()
                    if DEVICE == "cuda":
                        torch.cuda.synchronize()
                    end_bwd = timeit.default_timer()
                    bwd_times.append(end_bwd - start_bwd)
                
                # Optimizer step
                if run_optimizer and optimizer:
                    with nvtx.range("optimizer_step"):
                        start_optim = timeit.default_timer()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if DEVICE == "cuda":
                            torch.cuda.synchronize()
                        end_optim = timeit.default_timer()
                        optim_times.append(end_optim - start_optim)
                else:
                    model.zero_grad(set_to_none=True)
    
    return fwd_times, bwd_times, optim_times

def benchmark(
    cfg: Dict[str, int],
    batch: int,
    seq: int,
    warm: int,
    meas: int,
    forward_only: bool,
    run_optimizer: bool,
    profile_memory: bool = False,
    mixed_precision: bool = False,
    compiled: bool = False,
) -> Dict[str, float]:
    """
    Benchmark a transformer model with the given configuration.
    
    Args:
        cfg: Model configuration dictionary
        batch: Batch size
        seq: Sequence length
        warm: Number of warmup steps
        meas: Number of measurement steps
        forward_only: Whether to run only forward pass
        run_optimizer: Whether to run optimizer step
        profile_memory: Whether to profile memory usage
        mixed_precision: Whether to use mixed precision training
        compiled: Whether to use compiled version
        
    Returns:
        Dictionary with timing results
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(42)
    
    # Initialize model
    with nvtx.range("model_initialization"):
        model = Transformer(
            vocab_size=VOCAB_SIZE,
            context_length=seq,
            rope_theta=10000.0,
            **cfg
        ).to(DEVICE)
        
        if compiled:
            model = torch.compile(model)
            print("Using compiled model")
        
        model.train(not forward_only)
        
        # Create optimizer if needed
        optimizer = None
        if not forward_only and run_optimizer:
            optimizer = AdamW(
                model.parameters(), 
                lr=1e-4,
                weight_decay=0.01,
                betas=(0.9, 0.95)
            )
            print("Created AdamW optimizer")
        
        # Generate random batch
        x, y = random_batch(batch, seq)
    
    # Set up mixed precision if requested
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        print("Using mixed precision training")
    
    # Warmup phase
    with nvtx.range("warmup_phase"):
        print(f"Running {warm} warmup steps...")
        _ = run_steps(model, x, y, warm, not forward_only, run_optimizer, optimizer)
    
    # Start memory profiling if requested
    if profile_memory and DEVICE == "cuda":
        print("Starting memory profiling...")
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    # Measurement phase
    with nvtx.range("measurement_phase"):
        print(f"Running {meas} measurement steps...")
        fwd_times, bwd_times, optim_times = run_steps(
            model, x, y, meas, not forward_only, run_optimizer, optimizer
        )
    
    # Stop memory profiling and save snapshot if requested
    if profile_memory and DEVICE == "cuda":
        print("Saving memory snapshot...")
        os.makedirs("memory_snapshots", exist_ok=True)
        snapshot_name = f"memory_snapshots/{cfg['d_model']}_{seq}_{'forward' if forward_only else 'full'}_{'mixed' if mixed_precision else 'fp32'}.pickle"
        torch.cuda.memory._dump_snapshot(snapshot_name)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Memory snapshot saved to {snapshot_name}")
    
    # Calculate statistics
    results = {
        "fwd_mean": float(np.mean(fwd_times)),
        "fwd_std": float(np.std(fwd_times)),
    }
    
    if not forward_only:
        results["bwd_mean"] = float(np.mean(bwd_times))
        results["bwd_std"] = float(np.std(bwd_times))
        
        if run_optimizer:
            results["optim_mean"] = float(np.mean(optim_times))
            results["optim_std"] = float(np.std(optim_times))
            results["total_mean"] = results["fwd_mean"] + results["bwd_mean"] + results["optim_mean"]
        else:
            results["total_mean"] = results["fwd_mean"] + results["bwd_mean"]
    
    return results

def main() -> None:
    """Main function to parse arguments and run benchmarks."""
    parser = argparse.ArgumentParser(description="Profile Transformer models with Nsight Systems")
    parser.add_argument("--model-size", type=str, choices=list(MODEL_CONFIGS.keys()) + ["all"], required=True,
                       help="Model size to benchmark or 'all' for all sizes")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size to use")
    parser.add_argument("--seq-len", type=int, default=128,
                       help="Sequence length to use")
    parser.add_argument("--n-warmup", type=int, default=5,
                       help="Number of warmup steps")
    parser.add_argument("--n-measure", type=int, default=3,
                       help="Number of measurement steps")
    parser.add_argument("--forward-only", action="store_true",
                       help="Only profile forward pass")
    parser.add_argument("--run-optimizer", action="store_true",
                       help="Include optimizer step in profiling")
    parser.add_argument("--profile-memory", action="store_true",
                       help="Profile memory usage and save snapshot")
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--compiled", action="store_true",
                       help="Use compiled model")
    args = parser.parse_args()
    
    # Determine which model sizes to profile
    sizes = list(MODEL_CONFIGS.keys()) if args.model_size == "all" else [args.model_size]
    
    # Print header
    print(f"\n{'='*60}")
    print(f"Profiling with {'forward pass only' if args.forward_only else 'forward+backward'+(' and optimizer' if args.run_optimizer else '')}")
    print(f"Batch size: {args.batch_size}, Sequence length: {args.seq_len}")
    print(f"Warmup steps: {args.n_warmup}, Measurement steps: {args.n_measure}")
    print(f"Memory profiling: {'enabled' if args.profile_memory else 'disabled'}")
    print(f"Mixed precision: {'enabled' if args.mixed_precision else 'disabled'}")
    print(f"{'='*60}\n")
    
    # Run benchmarks for each model size
    for size in sizes:
        print(f"\nProfiling {size} model...")
        try:
            cfg = MODEL_CONFIGS[size]
            results = benchmark(
                cfg=cfg,
                batch=args.batch_size,
                seq=args.seq_len,
                warm=args.n_warmup,
                meas=args.n_measure,
                forward_only=args.forward_only,
                run_optimizer=args.run_optimizer,
                profile_memory=args.profile_memory,
                mixed_precision=args.mixed_precision,
                compiled=args.compiled
            )
            
            # Print results
            print(f"\nResults for {size} model:")
            print(f"  Forward pass: {results['fwd_mean']*1000:.2f} ± {results['fwd_std']*1000:.2f} ms")
            
            if not args.forward_only:
                print(f"  Backward pass: {results['bwd_mean']*1000:.2f} ± {results['bwd_std']*1000:.2f} ms")
                
                if args.run_optimizer:
                    print(f"  Optimizer step: {results['optim_mean']*1000:.2f} ± {results['optim_std']*1000:.2f} ms")
                    
                print(f"  Total step time: {results['total_mean']*1000:.2f} ms")
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Error: CUDA out of memory for {size} model with sequence length {args.seq_len}")
                print("Try using a smaller model or sequence length")
            else:
                print(f"Error: {str(e)}")
    
    print("\nProfiling complete.")
    if args.profile_memory:
        print("Memory snapshots are in the memory_snapshots directory.")
        print("View the results using PyTorch's memory visualization tool at https://pytorch.org/memory_viz")
    else:
        print("Analysis results are in the nsys profile output.")
        print("View the results using NVIDIA Nsight Systems desktop application.")

if __name__ == "__main__":
    main()