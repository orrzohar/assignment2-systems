#!/usr/bin/env python3
"""
Memory profiling for Transformer models

This script profiles memory usage of transformer models during forward pass,
backward pass, and optimizer steps using PyTorch's memory profiler.
"""
import argparse
import timeit
import pickle
from typing import Dict, Tuple, List, Optional
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.cuda.nvtx as nvtx
import pandas as pd

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
    print("Memory profiling requires a CUDA-capable GPU.")

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

def run_memory_profiling(
    model: Transformer,
    x: torch.Tensor,
    y: torch.Tensor,
    do_backward: bool,
    run_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    use_mixed_precision: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    output_file: str = "memory_snapshot.pickle"
) -> None:
    """
    Run memory profiling for a single step.
    
    Args:
        model: The transformer model
        x: Input tokens
        y: Target tokens
        do_backward: Whether to run backward pass
        run_optimizer: Whether to run optimizer step
        optimizer: Optimizer instance (required if run_optimizer is True)
        use_mixed_precision: Whether to use mixed precision
        dtype: Data type for mixed precision
        output_file: Path to save memory snapshot
    """
    criterion = torch.nn.CrossEntropyLoss()
    
    if run_optimizer and optimizer is None:
        raise ValueError("Optimizer must be provided if run_optimizer is True")
    
    # Create autocast context if using mixed precision
    autocast_context = torch.autocast('cuda', dtype=dtype) if use_mixed_precision else nullcontext()
    
    # Start recording memory history
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    try:
        # Forward pass
        with nvtx.range("forward_pass"):
            with autocast_context:
                logits = model(x)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
        
        # Backward pass
        if do_backward:
            with nvtx.range("backward_pass"):
                loss.backward()
            
            # Optimizer step
            if run_optimizer and optimizer:
                with nvtx.range("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)
        
        # Save memory snapshot
        torch.cuda.memory._dump_snapshot(output_file)
        print(f"Memory snapshot saved to {output_file}")
        
    finally:
        # Stop recording history
        torch.cuda.memory._record_memory_history(enabled=None)

def main() -> None:
    """Main function to parse arguments and run memory profiling."""
    parser = argparse.ArgumentParser(description="Profile memory usage of Transformer models")
    parser.add_argument("--model-size", type=str, choices=list(MODEL_CONFIGS.keys()), required=True,
                       help="Model size to profile")
    parser.add_argument("--seq-len", type=int, required=True,
                       help="Sequence length to use")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size to use")
    parser.add_argument("--forward-only", action="store_true",
                       help="Only profile forward pass")
    parser.add_argument("--run-optimizer", action="store_true",
                       help="Include optimizer step in profiling")
    parser.add_argument("--use-mixed-precision", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16",
                       help="Data type for mixed precision")
    parser.add_argument("--output-file", type=str, default="memory_snapshot.pickle",
                       help="Path to save memory snapshot")
    args = parser.parse_args()
    
    # Convert dtype string to torch.dtype
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(42)
    
    # Initialize model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        context_length=args.seq_len,
        rope_theta=10000.0,
        **MODEL_CONFIGS[args.model_size]
    ).to(DEVICE)
    
    model.train(not args.forward_only)
    
    # Create optimizer if needed
    optimizer = None
    if not args.forward_only and args.run_optimizer:
        optimizer = AdamW(
            model.parameters(), 
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        print("Created AdamW optimizer")
    
    # Generate random batch
    x, y = random_batch(args.batch_size, args.seq_len)
    
    # Run memory profiling
    run_memory_profiling(
        model,
        x,
        y,
        not args.forward_only,
        args.run_optimizer,
        optimizer,
        args.use_mixed_precision,
        dtype,
        args.output_file
    )

if __name__ == "__main__":
    main() 