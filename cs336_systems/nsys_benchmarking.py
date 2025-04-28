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
    use_mixed_precision: bool = False,
    dtype: torch.dtype = torch.bfloat16,
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
        use_mixed_precision: Whether to use mixed precision
        dtype: Data type for mixed precision (torch.bfloat16 or torch.float16)
        
    Returns:
        Tuple of (forward times, backward times, optimizer times)
    """
    fwd_times, bwd_times, optim_times = [], [], []
    criterion = torch.nn.CrossEntropyLoss()
    
    if run_optimizer and optimizer is None:
        raise ValueError("Optimizer must be provided if run_optimizer is True")
    
    # Create autocast context if using mixed precision
    autocast_context = torch.autocast(device_type=DEVICE, dtype=dtype) if use_mixed_precision else nullcontext()
    
    for i in range(n):
        # Start step with NVTX range
        with nvtx.range(f"step_{i}"):
            # Forward pass
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            
            with nvtx.range("forward_pass"):
                start_fwd = timeit.default_timer()
                with autocast_context:
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

def analyze_attention_components(
    model: Transformer,
    batch_size: int,
    seq_length: int,
    num_steps: int = 5
) -> Dict[str, float]:
    """
    Detailed analysis of attention layer components to compare softmax vs matrix multiplications.
    
    This function specifically profiles the three main components of the attention mechanism
    to compare their runtimes.
    
    Args:
        model: The transformer model
        batch_size: Batch size for input
        seq_length: Sequence length for input
        num_steps: Number of steps to run for statistical significance
        
    Returns:
        Dictionary with timing results for attention components
    """
    print(f"\nPerforming detailed analysis of attention layer components...")
    print(f"Model config: d_model={model.d_model}, num_heads={model.config['num_heads']}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_length}")
    
    # Generate random batch for the analysis
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_length), device=DEVICE, dtype=torch.long)
    
    # Create variables to store timing results
    attention_scores_times = []
    softmax_times = []
    final_matmul_times = []
    
    # Run the model a few times to get timing statistics
    model.eval()  # Set to evaluation mode since we don't need gradients
    
    for step in range(num_steps):
        with torch.no_grad(), nvtx.range(f"attention_analysis_step_{step}"):
            # Forward pass
            with nvtx.range("attention_analysis_forward"):
                # Force execution of each layer independently to get clean timings
                layer_outputs = []
                x_embeds = model.token_embeddings(x)
                for layer in model.layers:
                    # Synchronize before each layer to get clean timings
                    if DEVICE == "cuda":
                        torch.cuda.synchronize()
                    
                    # Process through the layer
                    layer_output = layer(x_embeds)
                    layer_outputs.append(layer_output)
                    x_embeds = layer_output
                    
                # Final normalization and output projection
                output = model.lm_head(model.ln_final(x_embeds))
    
    # Calculate theoretical FLOPs
    head_dim = model.d_model // model.config['num_heads']
    matmul_qk_flops = batch_size * model.config['num_heads'] * seq_length * seq_length * head_dim
    softmax_flops = batch_size * model.config['num_heads'] * seq_length * seq_length
    matmul_vo_flops = batch_size * model.config['num_heads'] * seq_length * seq_length * head_dim
    
    # Get average timings from the nsys profile (you'll need to look at this in the nsys GUI)
    # This function doesn't actually measure the times programmatically since nsys data 
    # isn't directly accessible from Python - you'll need to look at the nsys profile.
    # These lines are placeholders to remind you what to look for
    print(f"\nTo complete the attention component analysis:")
    print(f"1. Open the nsys profile and filter for 'attention_analysis_forward'")
    print(f"2. Within this range, look for the NVTX ranges:")
    print(f"   - computing_attention_scores (Q×K matrix multiplication)")
    print(f"   - computing_softmax (Softmax operation)")
    print(f"   - final_matmul (Attention weights × V matrix multiplication)")
    print(f"\n3. Compare their durations and FLOP counts:")
    print(f"   - Theoretical FLOP ratio for Q×K : Softmax : Attention×V = {head_dim} : 1 : {head_dim}")
    print(f"   - Theoretical FLOP counts:")
    print(f"     * Q×K matmul: {matmul_qk_flops:,} FLOPs")
    print(f"     * Softmax: {softmax_flops:,} FLOPs")
    print(f"     * Attention×V matmul: {matmul_vo_flops:,} FLOPs")
    print(f"\n4. Check if the runtime ratio matches the FLOP ratio")
    print(f"   - If not, what factors might explain the discrepancy?")
    
    # Return placeholder results - actual values should be retrieved from nsys
    return {
        "theory_qk_flops": matmul_qk_flops,
        "theory_softmax_flops": softmax_flops,
        "theory_av_flops": matmul_vo_flops,
        "head_dim": head_dim,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "num_heads": model.config['num_heads']
    }

def benchmark(
    cfg: Dict[str, int],
    batch: int,
    seq: int,
    warm: int,
    meas: int,
    forward_only: bool,
    run_optimizer: bool,
    use_mixed_precision: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    analyze_attention: bool = False,
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
        use_mixed_precision: Whether to use mixed precision
        dtype: Data type for mixed precision
        analyze_attention: Whether to perform detailed attention analysis
        
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
    
    # Perform attention component analysis if requested
    if analyze_attention:
        attention_results = analyze_attention_components(model, batch, seq)
    
    # Warmup phase
    with nvtx.range("warmup_phase"):
        print(f"Running {warm} warmup steps...")
        _ = run_steps(model, x, y, warm, not forward_only, run_optimizer, optimizer, use_mixed_precision, dtype)
    
    # Measurement phase
    with nvtx.range("measurement_phase"):
        print(f"Running {meas} measurement steps...")
        fwd_times, bwd_times, optim_times = run_steps(
            model, x, y, meas, not forward_only, run_optimizer, optimizer, use_mixed_precision, dtype
        )
    
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
            # Calculate total time statistics
            total_times = [f + b + o for f, b, o in zip(fwd_times, bwd_times, optim_times)]
            results["total_mean"] = float(np.mean(total_times))
            results["total_std"] = float(np.std(total_times))
        else:
            # Calculate total time statistics without optimizer
            total_times = [f + b for f, b in zip(fwd_times, bwd_times)]
            results["total_mean"] = float(np.mean(total_times))
            results["total_std"] = float(np.std(total_times))
    
    return results

def generate_latex_tables(results: Dict[str, Dict[str, float]], output_file: str) -> None:
    """
    Generate LaTeX tables from benchmarking results using pandas DataFrame.
    
    Args:
        results: Dictionary of results from benchmark runs
        output_file: Path to save the LaTeX tables
    """
    # Create DataFrame for combined table
    data = []
    
    for model_size in MODEL_CONFIGS.keys():
        if model_size in results:
            row = {'Model': model_size}
            
            # Forward pass data
            row['Forward Mean (ms)'] = results[model_size]['fwd_mean'] * 1000
            row['Forward Std (ms)'] = results[model_size]['fwd_std'] * 1000
            
            # Backward pass data
            if 'bwd_mean' in results[model_size]:
                row['Backward Mean (ms)'] = results[model_size]['bwd_mean'] * 1000
                row['Backward Std (ms)'] = results[model_size]['bwd_std'] * 1000
                
                # Total time data
                if 'total_mean' in results[model_size]:
                    row['Total Mean (ms)'] = results[model_size]['total_mean'] * 1000
                    row['Total Std (ms)'] = results[model_size]['total_std'] * 1000
            
            data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Generate LaTeX table
    latex_table = df.to_latex(
        index=False,
        float_format=lambda x: f'{x:.2f}',
        caption='Model Performance Timings',
        label='tab:model_timings',
        column_format='l|r|r|r|r|r|r'
    )
    
    # Write table to file
    with open(output_file, 'w') as f:
        f.write(latex_table)

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
    parser.add_argument("--use-mixed-precision", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16",
                       help="Data type for mixed precision")
    parser.add_argument("--output-tables", type=str, default="benchmark_tables.tex",
                       help="Path to save LaTeX tables")
    parser.add_argument("--analyze-attention", action="store_true",
                       help="Perform detailed analysis of attention components")
    args = parser.parse_args()
    
    # Convert dtype string to torch.dtype
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # Run benchmarks
    results = {}
    if args.model_size == "all":
        for size in MODEL_CONFIGS.keys():
            print(f"\nBenchmarking {size} model...")
            results[size] = benchmark(
                MODEL_CONFIGS[size],
                args.batch_size,
                args.seq_len,
                args.n_warmup,
                args.n_measure,
                args.forward_only,
                args.run_optimizer,
                args.use_mixed_precision,
                dtype,
                args.analyze_attention
            )
    else:
        results[args.model_size] = benchmark(
            MODEL_CONFIGS[args.model_size],
            args.batch_size,
            args.seq_len,
            args.n_warmup,
            args.n_measure,
            args.forward_only,
            args.run_optimizer,
            args.use_mixed_precision,
            dtype,
            args.analyze_attention
        )
    
    # Generate LaTeX tables
    generate_latex_tables(results, args.output_tables)
    print(f"\nLaTeX tables saved to {args.output_tables}")

if __name__ == "__main__":
    main()