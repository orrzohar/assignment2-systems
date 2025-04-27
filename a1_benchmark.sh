#!/bin/bash

# Script to run benchmarking experiments for CS336 Assignment 2, Section 1.1.3

echo "Starting benchmarking script..."

# --- Configuration ---
PYTHON_SCRIPT="cs336_systems/nsys_benchmarking.py" # Path to your benchmarking script
BATCH_SIZE=4
SEQ_LEN=128
N_MEASURE=10
OUTPUT_DIR="./benchmark_results" # Directory to save results

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if the python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Benchmarking script '$PYTHON_SCRIPT' not found."
    exit 1
fi

# --- Experiment (b): Standard Warmup (5 steps) ---
echo "--- Running Experiment (b): 5 Warmup Steps ---"
N_WARMUP_B=5

python "$PYTHON_SCRIPT" \
    --model-size all \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --n-warmup $N_WARMUP_B \
    --n-measure $N_MEASURE

if [ $? -ne 0 ]; then
    echo "Error running experiment (b). Exiting."
    exit 1
fi
echo "Experiment (b) complete."
echo "-------------------------------------------------"


# --- Experiment (c): Varying Warmup Steps (0, 1, 2 steps) ---
echo "--- Running Experiment (c): Varying Warmup Steps ---"

for n_warmup_c in 0 1 2; do
    echo "Running with $n_warmup_c Warmup Steps..."
    
    python "$PYTHON_SCRIPT" \
        --model-size all \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --n-warmup $n_warmup_c \
        --n-measure $N_MEASURE

    if [ $? -ne 0 ]; then
        echo "Error running experiment (c) with $n_warmup_c warmup steps. Continuing..."
    else
        echo "Experiment (c) with $n_warmup_c warmup steps complete."
    fi
    echo "----------------------------"
done

echo "--- All Benchmarking Runs Complete ---"

exit 0
