#!/usr/bin/env python3
"""
Memory profiling for Transformer models

This script profiles memory usage of Transformer models during various stages
of training and inference using PyTorch's memory profiler.
"""
import argparse
import os
from typing import Optional
import torch
from torch.profiler import profile, ProfilerActivity
from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.optimizer import AdamW

# Configuration
VOCAB_SIZE = 10_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")
else:
    print(f"Warning: CUDA is not available. Using CPU instead.")
    print("Memory profiling requires a CUDA-capable GPU.")

# Model configurations based on Table 1 in the assignment
MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def run_memory_profiling(
    model_size: str,
    seq_len: int,
    batch_size: int,
    output_dir: str,
    forward_only: bool = False,
    use_mixed_precision: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Run memory profiling for a Transformer model.
    
    Args:
        model_size: Size of model to profile
        seq_len: Sequence length
        batch_size: Batch size
        output_dir: Directory to save profiling results
        forward_only: Whether to profile only forward pass
        use_mixed_precision: Whether to use mixed precision
        dtype: Data type for mixed precision
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        context_length=seq_len,
        rope_theta=10000.0,
        **MODEL_CONFIGS[model_size]
    ).to(DEVICE)
    
    model.train(not forward_only)
    
    # Create optimizer if needed
    optimizer = None
    if not forward_only:
        optimizer = AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
    
    # Create random batch
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=DEVICE, dtype=torch.long)
    y = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=DEVICE, dtype=torch.long)
    
    # Create autocast context if using mixed precision
    autocast_context = torch.autocast(device_type=DEVICE, dtype=dtype) if use_mixed_precision else torch.cuda.amp.autocast(enabled=False)
    
    # Start recording memory history
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    # Run profiling
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # Forward pass
        with autocast_context:
            logits = model(x)
        
        if not forward_only:
            # Backward pass
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            loss.backward()
            
            # Optimizer step
            if optimizer:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        prof.step()
    
    # Save profiling results
    prefix = f"{model_size}_seq{seq_len}_batch{batch_size}"
    if forward_only:
        prefix += "_forward"
    else:
        prefix += "_full"
    if use_mixed_precision:
        prefix += f"_{dtype}"
    
    # Save memory timeline
    prof.export_memory_timeline(f"{output_dir}/{prefix}_timeline.html", device=DEVICE)
    
    # Save memory snapshot
    torch.cuda.memory._dump_snapshot(f"{output_dir}/{prefix}_snapshot.pickle")
    
    # Stop recording history
    torch.cuda.memory._record_memory_history(enabled=None)
    
    print(f"Memory profiling completed. Results saved to {output_dir}/")
    print(f"- Timeline: {prefix}_timeline.html")
    print(f"- Snapshot: {prefix}_snapshot.pickle")

def main() -> None:
    """Main function to parse arguments and run memory profiling."""
    parser = argparse.ArgumentParser(description="Profile memory usage of Transformer models")
    parser.add_argument("--model-size", type=str, choices=list(MODEL_CONFIGS.keys()), default="2.7B",
                       help="Model size to profile")
    parser.add_argument("--seq-len", type=int, default=128,
                       help="Sequence length to use")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size to use")
    parser.add_argument("--output-dir", type=str, default="memory_profiles",
                       help="Directory to save profiling results")
    parser.add_argument("--forward-only", action="store_true",
                       help="Only profile forward pass")
    parser.add_argument("--use-mixed-precision", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16",
                       help="Data type for mixed precision")
    args = parser.parse_args()
    
    # Convert dtype string to torch.dtype
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    run_memory_profiling(
        model_size=args.model_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        forward_only=args.forward_only,
        use_mixed_precision=args.use_mixed_precision,
        dtype=dtype
    )

if __name__ == "__main__":
    main() 