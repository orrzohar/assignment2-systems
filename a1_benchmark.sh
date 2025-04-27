#!/bin/bash

# Script to run benchmarking experiments for CS336 Assignment 2, Section 1.1.3

echo "Starting benchmarking script..."

# --- Configuration ---
PYTHON_SCRIPT="cs336_systems/benchmarking.py" # Path to your benchmarking script
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
CSV_OUT_B="$OUTPUT_DIR/results_warmup_${N_WARMUP_B}.csv"
LATEX_OUT_B="$OUTPUT_DIR/results_warmup_${N_WARMUP_B}.tex"

# Run for all model sizes, including backward pass
python "$PYTHON_SCRIPT" \
    --model-size all \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --n-warmup $N_WARMUP_B \
    --n-measure $N_MEASURE \
    --csv-out "$CSV_OUT_B" \
    --latex-out "$LATEX_OUT_B"

if [ $? -ne 0 ]; then
    echo "Error running experiment (b). Exiting."
    exit 1
fi
echo "Experiment (b) complete. Results saved to $CSV_OUT_B and $LATEX_OUT_B"
echo "-------------------------------------------------"


# --- Experiment (c): Varying Warmup Steps (0, 1, 2 steps) ---
echo "--- Running Experiment (c): Varying Warmup Steps ---"

for n_warmup_c in 0 1 2; do
    echo "Running with $n_warmup_c Warmup Steps..."
    CSV_OUT_C="$OUTPUT_DIR/results_warmup_${n_warmup_c}.csv"
    LATEX_OUT_C="$OUTPUT_DIR/results_warmup_${n_warmup_c}.tex"

    # Run for all model sizes, including backward pass
    python "$PYTHON_SCRIPT" \
        --model-size all \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --n-warmup $n_warmup_c \
        --n-measure $N_MEASURE \
        --csv-out "$CSV_OUT_C" \
        --latex-out "$LATEX_OUT_C"

    if [ $? -ne 0 ]; then
        echo "Error running experiment (c) with $n_warmup_c warmup steps. Continuing..."
    else
        echo "Experiment (c) with $n_warmup_c warmup steps complete. Results saved to $CSV_OUT_C and $LATEX_OUT_C"
    fi
    echo "----------------------------"
done

echo "--- All Benchmarking Runs Complete ---"

exit 0
