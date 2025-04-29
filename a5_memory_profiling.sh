#!/bin/bash

# Script to run memory profiling experiments

echo "Starting memory profiling..."

# Create output directory
mkdir -p memory_profiles

# Model size to test (2.7B as specified in the problem)
MODEL_SIZE="2.7B"

# Sequence lengths to test
SEQ_LENGTHS=(128 256 512)

# Batch size
BATCH_SIZE=4

# Run profiling for each sequence length
for seq_len in "${SEQ_LENGTHS[@]}"; do
    echo "=============================================================="
    echo "Profiling sequence length: $seq_len"
    echo "=============================================================="
    
    # Forward pass only
    echo "Running forward pass only..."
    python -m cs336_systems.memory_profiling \
        --model-size "$MODEL_SIZE" \
        --seq-len "$seq_len" \
        --batch-size "$BATCH_SIZE" \
        --forward-only \
        --output-file "memory_profiles/2.7B_seq${seq_len}_forward.pickle"
    
    # Full training step
    echo "Running full training step..."
    python -m cs336_systems.memory_profiling \
        --model-size "$MODEL_SIZE" \
        --seq-len "$seq_len" \
        --batch-size "$BATCH_SIZE" \
        --run-optimizer \
        --output-file "memory_profiles/2.7B_seq${seq_len}_full_training.pickle"
    
    # Mixed precision (BF16)
    echo "Running mixed precision (BF16)..."
    python -m cs336_systems.memory_profiling \
        --model-size "$MODEL_SIZE" \
        --seq-len "$seq_len" \
        --batch-size "$BATCH_SIZE" \
        --use-mixed-precision \
        --dtype bfloat16 \
        --run-optimizer \
        --output-file "memory_profiles/2.7B_seq${seq_len}_bf16.pickle"
    
    echo "Completed profiling for sequence length: $seq_len"
    echo ""
done

echo "Memory profiling completed. Results are in the memory_profiles directory."
echo "To analyze the results:"
echo "1. Go to https://pytorch.org/memory_viz"
echo "2. Drag and drop the .pickle files onto the page"
echo "3. Use the 'Detail' slider to adjust the level of detail"
echo "4. Look for the 'Active Memory Timeline' to see memory usage over time" 