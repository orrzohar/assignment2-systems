#!/bin/bash

# Create output directory
mkdir -p nsys_profiles

# Model sizes to test
MODEL_SIZES=("small" "medium" "large" "xl" "2.7B")

# Sequence lengths to test
SEQ_LENGTHS=(128 256 512 1024)

# Run profiling for each model size and sequence length
for model_size in "${MODEL_SIZES[@]}"; do
    for seq_len in "${SEQ_LENGTHS[@]}"; do
        echo "Profiling model size: $model_size, sequence length: $seq_len"
        
        # Run forward pass only
        nsys profile -o "nsys_profiles/${model_size}_seq${seq_len}_forward" \
            python -m cs336_systems.nsys_benchmarking \
            --model-size "$model_size" \
            --seq-len "$seq_len" \
            --forward-only
        
        # Run full training step (forward + backward)
        nsys profile -o "nsys_profiles/${model_size}_seq${seq_len}_full" \
            python -m cs336_systems.nsys_benchmarking \
            --model-size "$model_size" \
            --seq-len "$seq_len"
    done
done

echo "Profiling completed. Results are in the nsys_profiles directory."
echo "You can view the results using NVIDIA Nsight Systems desktop application." 