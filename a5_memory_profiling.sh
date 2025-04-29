#!/bin/bash

# Script to run memory profiling experiments for CS336 Assignment 2, Section 1.1.6

echo "Starting memory profiling script..."

# --- Configuration ---
PYTHON_SCRIPT="cs336_systems/nsys_benchmarking.py"
BATCH_SIZE=4
N_WARMUP=5
N_MEASURE=3
OUTPUT_DIR="memory_snapshots"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if the python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Benchmarking script '$PYTHON_SCRIPT' not found."
    exit 1
fi

# --- Task (a): Profile 2.7B model with different context lengths ---
echo "--- Running Task (a): Profile 2.7B model with different context lengths ---"

# Context lengths to test
SEQ_LENGTHS=(128 256 512)

# Forward pass only
echo "Profiling forward pass only..."
for seq_len in "${SEQ_LENGTHS[@]}"; do
    echo "Running with sequence length: $seq_len"
    python "$PYTHON_SCRIPT" \
        --model-size 2.7B \
        --batch-size $BATCH_SIZE \
        --seq-len $seq_len \
        --n-warmup $N_WARMUP \
        --n-measure $N_MEASURE \
        --forward-only \
        --profile-memory
done

# Full training step
echo "Profiling full training step..."
for seq_len in "${SEQ_LENGTHS[@]}"; do
    echo "Running with sequence length: $seq_len"
    python "$PYTHON_SCRIPT" \
        --model-size 2.7B \
        --batch-size $BATCH_SIZE \
        --seq-len $seq_len \
        --n-warmup $N_WARMUP \
        --n-measure $N_MEASURE \
        --run-optimizer \
        --profile-memory
done

# --- Task (c): Profile with mixed precision ---
echo "--- Running Task (c): Profile with mixed precision ---"

# Forward pass only with mixed precision
echo "Profiling forward pass only with mixed precision..."
for seq_len in "${SEQ_LENGTHS[@]}"; do
    echo "Running with sequence length: $seq_len"
    python "$PYTHON_SCRIPT" \
        --model-size 2.7B \
        --batch-size $BATCH_SIZE \
        --seq-len $seq_len \
        --n-warmup $N_WARMUP \
        --n-measure $N_MEASURE \
        --forward-only \
        --profile-memory \
        --mixed-precision
done

# Full training step with mixed precision
echo "Profiling full training step with mixed precision..."
for seq_len in "${SEQ_LENGTHS[@]}"; do
    echo "Running with sequence length: $seq_len"
    python "$PYTHON_SCRIPT" \
        --model-size 2.7B \
        --batch-size $BATCH_SIZE \
        --seq-len $seq_len \
        --n-warmup $N_WARMUP \
        --n-measure $N_MEASURE \
        --run-optimizer \
        --profile-memory \
        --mixed-precision
done

echo "--- All Memory Profiling Runs Complete ---"
echo "Memory snapshots are in the $OUTPUT_DIR directory."
echo "View the results using PyTorch's memory visualization tool at https://pytorch.org/memory_viz"

exit 0 