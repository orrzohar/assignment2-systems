import argparse
import timeit
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_systems.model_annotated import annotated_scaled_dot_product_attention

# Replace the original attention function with the annotated version
Transformer.scaled_dot_product_attention = annotated_scaled_dot_product_attention

VOCAB_SIZE = 10_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def random_batch(batch: int, seq: int, d_model: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, VOCAB_SIZE, (batch, seq), device=DEVICE)
    y = torch.randn(batch, seq, d_model, device=DEVICE, dtype=torch.float32)
    return x, y

def run_steps(
    model: Transformer,
    x: torch.Tensor,
    y: torch.Tensor,
    n: int,
    do_backward: bool,
) -> Tuple[List[float], List[float]]:
    fwd_times, full_times = [], []

    for _ in range(n):
        with nvtx.range("forward_pass"):
            start = timeit.default_timer()
            out = model(x)
            torch.cuda.synchronize()
            mid = timeit.default_timer()

        if do_backward:
            with nvtx.range("backward_pass"):
                loss = torch.nn.functional.mse_loss(out, y)
                loss.backward()
                torch.cuda.synchronize()
                end = timeit.default_timer()
                full_times.append(end - start)
                model.zero_grad(set_to_none=True)
        fwd_times.append(mid - start)

    return fwd_times, full_times if do_backward else fwd_times

def benchmark(
    cfg: Dict[str, int],
    batch: int,
    seq: int,
    warm: int,
    meas: int,
    forward_only: bool,
) -> Tuple[float, float, float]:
    torch.manual_seed(42)
    with nvtx.range("model_initialization"):
        model = Transformer(
            vocab_size=VOCAB_SIZE,
            context_length=seq,
            rope_theta=10000.0,
            **cfg
        ).to(DEVICE).train()
        x, y = random_batch(batch, seq, cfg["d_model"])

    # warm-up
    with nvtx.range("warmup"):
        run_steps(model, x, y, warm, not forward_only)

    # measurement
    with nvtx.range("measurement"):
        fwd, full = run_steps(model, x, y, meas, not forward_only)

    fwd_mean = float(np.mean(fwd))
    fwd_std  = float(np.std(fwd))
    if forward_only:
        return fwd_mean, fwd_std, 0.0

    full_mean = float(np.mean(full))
    bwd_mean = full_mean - fwd_mean
    return fwd_mean, fwd_std, bwd_mean

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", choices=list(MODEL_CONFIGS) + ["all"], required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len",   type=int, default=128)
    p.add_argument("--n-warmup",  type=int, default=5)
    p.add_argument("--n-measure", type=int, default=10)
    p.add_argument("--forward-only", action="store_true")
    args = p.parse_args()

    sizes = MODEL_CONFIGS.keys() if args.model_size == "all" else [args.model_size]

    header = "size | pass | mean(ms) | std(ms)" if not args.forward_only else "size | fwd mean(ms) | fwd std(ms)"
    print(header)
    print("-" * len(header))

    for s in sizes:
        cfg = MODEL_CONFIGS[s]
        fwd_mean, fwd_std, bwd_mean = benchmark(
            cfg,
            args.batch_size,
            args.seq_len,
            args.n_warmup,
            args.n_measure,
            args.forward_only,
        )

        if args.forward_only:
            print(f"{s:6} | {fwd_mean*1e3:9.2f} | {fwd_std*1e3:8.2f}")
        else:
            print(f"{s:6} | fwd  | {fwd_mean*1e3:8.2f} | {fwd_std*1e3:8.2f}")
            print(f"{'':6} | bwd  | {bwd_mean*1e3:8.2f} |    –    ")

if __name__ == "__main__":
    main() 