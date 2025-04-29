#!/bin/bash

# Script to run mixed precision benchmarking experiments
echo "=============================================================="
echo "Casting data types: float16"
python cs336_systems/mixed_precision_toy_model.py
echo "============================="

echo "Casting data types: bfloat16"
python cs336_systems/mixed_precision_toy_model.py --precision bfloat16
echo "=============================================================="

echo "Starting mixed precision benchmarking..."

# Create output directory
mkdir -p mixed_precision_results

# Model sizes to test
MODEL_SIZES=("small" "medium" "large" "xl" "2.7B")

# Sequence length (using same as assignment 1)
SEQ_LEN=128

# Batch size
BATCH_SIZE=4

# Number of steps
WARMUP_STEPS=2
MEASURE_STEPS=3

# Run benchmarks for each model size
for model_size in "${MODEL_SIZES[@]}"; do
    echo "=============================================================="
    echo "Benchmarking model size: $model_size"
    echo "=============================================================="
    
    # Full precision (FP32)
    echo "Running full precision (FP32)..."
    python -m cs336_systems.nsys_benchmarking \
        --model-size "$model_size" \
        --seq-len "$SEQ_LEN" \
        --batch-size "$BATCH_SIZE" \
        --n-warmup "$WARMUP_STEPS" \
        --n-measure "$MEASURE_STEPS" \
        --output-tables "mixed_precision_results/${model_size}_fp32.tex"
    
    # Mixed precision with BF16
    echo "Running mixed precision with BF16..."
    python -m cs336_systems.nsys_benchmarking \
        --model-size "$model_size" \
        --seq-len "$SEQ_LEN" \
        --batch-size "$BATCH_SIZE" \
        --n-warmup "$WARMUP_STEPS" \
        --n-measure "$MEASURE_STEPS" \
        --use-mixed-precision \
        --dtype bfloat16 \
        --output-tables "mixed_precision_results/${model_size}_bf16.tex"
    
    echo "Completed benchmarking for model size: $model_size"
    echo ""
done

echo "Mixed precision benchmarking completed. Results are in the mixed_precision_results directory." 