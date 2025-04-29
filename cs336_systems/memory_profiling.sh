#!/bin/bash

# Create output directory
mkdir -p memory_profiles

# Set model size and sequence lengths to test
MODEL_SIZE="2.7B"
SEQ_LENS=(128 256 512)
BATCH_SIZE=4

# Run profiling for each sequence length
for SEQ_LEN in "${SEQ_LENS[@]}"; do
    echo "Profiling sequence length $SEQ_LEN..."
    
    # Forward pass only
    echo "Running forward pass..."
    python cs336_systems/memory_profiling.py \
        --model-size "$MODEL_SIZE" \
        --seq-len "$SEQ_LEN" \
        --batch-size "$BATCH_SIZE" \
        --output-dir memory_profiles \
        --forward-only
    
    # Full training step
    echo "Running full training step..."
    python cs336_systems/memory_profiling.py \
        --model-size "$MODEL_SIZE" \
        --seq-len "$SEQ_LEN" \
        --batch-size "$BATCH_SIZE" \
        --output-dir memory_profiles
    
    # Mixed precision (BF16)
    echo "Running mixed precision (BF16)..."
    python cs336_systems/memory_profiling.py \
        --model-size "$MODEL_SIZE" \
        --seq-len "$SEQ_LEN" \
        --batch-size "$BATCH_SIZE" \
        --output-dir memory_profiles \
        --use-mixed-precision \
        --dtype bfloat16
done

echo "Memory profiling completed. Results saved to memory_profiles/"
echo "You can analyze the results using the PyTorch memory visualization tool at https://pytorch.org/memory_viz"
echo "Just drag and drop the .pickle files and adjust the detail level to analyze memory usage patterns." 