#!/usr/bin/env python3
# cs336_systems/benchmarking.py
import argparse
import timeit
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
# Ensure cs336_basics is installed or available in the path
# If using the provided structure, it should be importable
try:
    from cs336_basics.model import BasicsTransformerLM as Transformer
    # --- NVTX Annotation Point ---
    # If you create an annotated attention function as suggested in 1.1.4,
    # import and assign it here. Example:
    # from cs336_basics.model_annotated import annotated_scaled_dot_product_attention
    # from cs336_basics import model as basics_model
    # basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    # print("Using annotated attention function.")
except ImportError:
    print("Error: Could not import BasicsTransformerLM.")
    print("Please ensure the cs336_basics package is installed or accessible.")
    print("You might need to run `pip install -e ./cs336-basics` from the assignment root directory.")
    exit(1)
except AttributeError:
    print("Note: Could not find or assign annotated attention. Using default.")
    pass # Continue without annotated attention if not found

# Import NVTX if on CUDA device
if torch.cuda.is_available():
    try:
        import torch.cuda.nvtx as nvtx
        print("NVTX imported successfully.")
    except ImportError:
        print("Warning: torch.cuda.nvtx not found. NVTX ranges will be disabled.")
        nvtx = None
else:
    nvtx = None # NVTX is only for CUDA

# --- Configuration ---
VOCAB_SIZE = 10_000
# Determine device and print info
if torch.cuda.is_available():
    DEVICE = "cuda"
    try:
        print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    except Exception as e:
        print(f"Using device: {DEVICE}, but failed to get device name: {e}")
else:
    DEVICE = "cpu"
    print(f"Using device: {DEVICE}")

# Model configurations based on Table 1
MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

# --- Helper Functions ---

def random_batch(batch: int, seq: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a random batch of input and target tensors."""
    x = torch.randint(0, VOCAB_SIZE, (batch, seq), device=DEVICE, dtype=torch.long)
    y = torch.randint(0, VOCAB_SIZE, (batch, seq), device=DEVICE, dtype=torch.long)
    return x, y

def run_steps(
    model: Transformer,
    x: torch.Tensor,
    y: torch.Tensor,
    n: int,
    do_backward: bool,
    optimizer: Optional[torch.optim.Optimizer] = None, # Add optimizer argument
    run_optimizer_step: bool = False, # Flag to run optimizer.step()
) -> Tuple[List[float], List[float]]:
    """
    Runs the model for 'n' steps, timing the forward, optionally backward,
    and optionally optimizer steps. Includes NVTX ranges for profiling.

    Args:
        model: The Transformer model.
        x: Input tensor.
        y: Target tensor.
        n: Number of steps to run.
        do_backward: Whether to perform the backward pass and time it.
        optimizer: Optional optimizer instance. Required if run_optimizer_step is True.
        run_optimizer_step: Whether to run optimizer.step().

    Returns:
        A tuple containing lists of forward pass times and full step times (fwd+bwd+optim).
        Full step times list will be empty if do_backward is False.
    """
    fwd_times, full_times = [], []
    criterion = torch.nn.CrossEntropyLoss()

    if run_optimizer_step and optimizer is None:
        raise ValueError("Optimizer must be provided if run_optimizer_step is True.")

    for i in range(n):
        # --- NVTX Range: Measurement Step ---
        # Note: NVTX ranges only work on CUDA
        step_range = nvtx.range(f"step_{i}") if nvtx and DEVICE == "cuda" else None
        if step_range: step_range.__enter__()

        # Synchronize before starting the timer for accurate GPU timing
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = timeit.default_timer()

        # --- NVTX Range: Forward Pass ---
        fwd_range = nvtx.range("forward_pass") if nvtx and DEVICE == "cuda" else None
        if fwd_range: fwd_range.__enter__()

        logits = model(x)

        if fwd_range: fwd_range.__exit__(None, None, None)
        # --- End NVTX Range: Forward Pass ---

        # Synchronize after the forward pass
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = timeit.default_timer()
        fwd_times.append(t1 - t0) # Record forward time

        if do_backward:
            # Calculate loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            # --- NVTX Range: Backward Pass ---
            bwd_range = nvtx.range("backward_pass") if nvtx and DEVICE == "cuda" else None
            if bwd_range: bwd_range.__enter__()

            loss.backward()

            if bwd_range: bwd_range.__exit__(None, None, None)
            # --- End NVTX Range: Backward Pass ---

            # Synchronize after the backward pass for timing t2 accurately
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t2 = timeit.default_timer()

            if run_optimizer_step:
                # --- NVTX Range: Optimizer Step ---
                opt_range = nvtx.range("optimizer_step") if nvtx and DEVICE == "cuda" else None
                if opt_range: opt_range.__enter__()

                optimizer.step() # Run optimizer step

                if opt_range: opt_range.__exit__(None, None, None)
                # --- End NVTX Range: Optimizer Step ---

                # Synchronize after the optimizer step for timing t3 accurately
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                t3 = timeit.default_timer()
                full_times.append(t3 - t0) # Record full step time (fwd + bwd + optim)

                # Zero gradients AFTER optimizer step
                optimizer.zero_grad(set_to_none=True)

            else:
                 # If only fwd+bwd (no optimizer step)
                 full_times.append(t2 - t0) # Record fwd + bwd time
                 # Zero gradients for the next iteration
                 model.zero_grad(set_to_none=True)

        else: # Only forward pass
             model.zero_grad(set_to_none=True) # Still good practice to zero

        if step_range: step_range.__exit__(None, None, None)
        # --- End NVTX Range: Measurement Step ---

    return fwd_times, full_times

def benchmark(
    cfg: Dict[str, int],
    batch: int,
    seq: int,
    warm: int,
    meas: int,
    forward_only: bool,
    run_optimizer: bool, # Add flag to control optimizer run
) -> Tuple[float, float, float, float, float]:
    """
    Benchmarks a given model configuration, optionally including optimizer step.

    Args:
        cfg: Dictionary containing model hyperparameters (d_model, etc.).
        batch: Batch size.
        seq: Sequence length.
        warm: Number of warm-up steps.
        meas: Number of measurement steps.
        forward_only: If True, only benchmark the forward pass.
        run_optimizer: If True, include optimizer step in backward pass timing.

    Returns:
        A tuple containing:
        - Mean forward pass time (seconds).
        - Standard deviation of forward pass time (seconds).
        - Mean backward pass time (seconds, 0 if forward_only).
        - Standard deviation of backward pass time (seconds, nan if forward_only).
        - Mean optimizer step time (seconds, 0 if not run_optimizer).
          Note: Backward time includes optimizer time if run_optimizer is True.
                This function returns the isolated optimizer time estimate.
    """
    # Set seed for reproducibility
    torch.manual_seed(42)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(42) # Set seed for all GPUs if applicable

    # Initialize the model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        context_length=seq,
        rope_theta=10000.0, # Default theta, adjust if needed
        **cfg,
    ).to(DEVICE)

    # Initialize optimizer if needed
    optimizer = None
    if not forward_only and run_optimizer:
        # Using AdamW as a common example
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        print("Optimizer (AdamW) initialized.")

    # Set model to training mode if backward pass is needed, else evaluation mode
    model.train(not forward_only)

    # Generate a random batch
    x, y = random_batch(batch, seq)

    # --- NVTX Range: Warmup ---
    # We wrap the whole warmup phase to potentially exclude it easily in Nsight Systems
    warmup_range = nvtx.range("warmup_phase") if nvtx and DEVICE == "cuda" else None
    if warmup_range: warmup_range.__enter__()

    print(f"Running {warm} warm-up steps...")
    # Run warm-up steps (discard timings). Include optimizer if it will be measured.
    _ = run_steps(model, x, y, warm, not forward_only, optimizer, run_optimizer)
    print("Warm-up complete.")

    if warmup_range: warmup_range.__exit__(None, None, None)
    # --- End NVTX Range: Warmup ---


    # --- NVTX Range: Measurement ---
    measurement_range = nvtx.range("measurement_phase") if nvtx and DEVICE == "cuda" else None
    if measurement_range: measurement_range.__enter__()

    print(f"Running {meas} measurement steps...")
    fwd_times, full_times = run_steps(model, x, y, meas, not forward_only, optimizer, run_optimizer)
    print("Measurement complete.")

    if measurement_range: measurement_range.__exit__(None, None, None)
    # --- End NVTX Range: Measurement ---


    # Calculate statistics for forward pass
    fwd_mean = float(np.mean(fwd_times))
    fwd_std = float(np.std(fwd_times))

    # Calculate statistics for backward/full pass
    bwd_mean = 0.0
    bwd_std = np.nan
    optim_mean = 0.0 # Store optimizer time separately if calculated

    if not forward_only:
        if not full_times:
             print("Warning: No full step times recorded despite do_backward=True.")
        else:
            full_mean = float(np.mean(full_times))
            # Calculate backward+optimizer times
            bwd_optim_times = [full - fwd for full, fwd in zip(full_times, fwd_times)]
            bwd_mean = float(np.mean(bwd_optim_times)) # This is bwd+optim if run_optimizer=True
            bwd_std = float(np.std(bwd_optim_times))

            # If optimizer was run, estimate its time by comparing runs
            # Note: This requires running benchmark twice (once with, once without optimizer)
            # for an accurate breakdown. For simplicity here, we report bwd+optim as 'bwd'.
            # A more accurate approach would involve timing optimizer separately or
            # analyzing NVTX ranges directly in Nsight Systems.
            # We will report bwd+optim time as 'bwd' phase time, and optim_mean as 0 here.
            # The real analysis for part (d) and (e) should come from Nsight Systems GUI.


    return fwd_mean, fwd_std, bwd_mean, bwd_std, optim_mean

# --- Main Execution ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Transformer models with different configurations")
    parser.add_argument("--model_size", type=str, choices=list(MODEL_CONFIGS.keys()), required=True,
                      help="Model size to benchmark")
    parser.add_argument("--context_length", type=int, required=True,
                      help="Context length to use")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size to use")
    parser.add_argument("--warmup_steps", type=int, default=10,
                      help="Number of warmup steps")
    parser.add_argument("--measurement_steps", type=int, default=100,
                      help="Number of measurement steps")
    parser.add_argument("--forward_only", action="store_true",
                      help="Only run forward pass")
    parser.add_argument("--run_optimizer", action="store_true",
                      help="Include optimizer step in measurements")
    parser.add_argument("--output_file", type=str, default="benchmark_results.csv",
                      help="Output file for results")

    args = parser.parse_args()

    # Import and set up annotated attention if available
    try:
        from cs336_systems.model_annotated import annotated_scaled_dot_product_attention
        from cs336_basics import model as basics_model
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
        print("Using annotated attention function for profiling")
    except ImportError:
        print("Note: Could not import annotated attention. Using default implementation")

    # Run benchmark
    cfg = MODEL_CONFIGS[args.model_size]
    print(f"\nBenchmarking {args.model_size} model with context length {args.context_length}")
    print(f"Configuration: {cfg}")
    
    try:
        fwd_mean, fwd_std, bwd_mean, bwd_std, optim_mean = benchmark(
            cfg=cfg,
            batch=args.batch_size,
            seq=args.context_length,
            warm=args.warmup_steps,
            meas=args.measurement_steps,
            forward_only=args.forward_only,
            run_optimizer=args.run_optimizer
        )

        # Print results
        print("\nResults:")
        print(f"Forward pass: {fwd_mean:.4f} ± {fwd_std:.4f} seconds")
        if not args.forward_only:
            print(f"Backward pass: {bwd_mean:.4f} ± {bwd_std:.4f} seconds")
            if args.run_optimizer:
                print(f"Optimizer step: {optim_mean:.4f} seconds")

        # Save results to CSV
        results = {
            "model_size": args.model_size,
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "forward_mean": fwd_mean,
            "forward_std": fwd_std,
            "backward_mean": bwd_mean,
            "backward_std": bwd_std,
            "optimizer_mean": optim_mean
        }
        
        df = pd.DataFrame([results])
        df.to_csv(args.output_file, mode='a', header=not pd.io.common.file_exists(args.output_file), index=False)
        print(f"\nResults saved to {args.output_file}")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\nError: Out of memory for {args.model_size} model with context length {args.context_length}")
            print("Please try with a smaller model or context length")
        else:
            print(f"\nError during benchmarking: {e}")

if __name__ == "__main__":
    main()
