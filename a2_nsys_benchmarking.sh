#!/bin/bash
# Advanced profiling script for CS336 Assignment 2, Part 1.1.4
# This script runs nsys profiling on Transformer models with different 
# configurations and captures NVTX annotations for detailed analysis

# Create output directory
mkdir -p nsys_profiles

# Model sizes to test (from Table 1 in the assignment)
# Comment out larger models if you're running out of memory
MODEL_SIZES=("small" "medium" "large" "xl" "2.7B")

# Sequence lengths to test
SEQ_LENGTHS=(128 256 512 1024)

# Batch size (keep small to avoid OOM)
BATCH_SIZE=4

# Number of steps for warmup and measurement
WARMUP_STEPS=2
MEASURE_STEPS=3

# Simplified nsys options that should work across versions
NSYS_OPTIONS="--trace=cuda,nvtx --stats=true"

echo "Starting NVIDIA Nsight Systems profiling for Transformer models"

# Run profiling for each model size and sequence length
for model_size in "${MODEL_SIZES[@]}"; do
    for seq_len in "${SEQ_LENGTHS[@]}"; do
        echo "=============================================================="
        echo "Profiling model size: $model_size, sequence length: $seq_len"
        echo "=============================================================="
        
        # Forward pass only
        nsys profile $NSYS_OPTIONS \
            -o "nsys_profiles/${model_size}_seq${seq_len}_forward" \
            python -m cs336_systems.nsys_benchmarking \
            --model-size "$model_size" \
            --seq-len "$seq_len" \
            --batch-size "$BATCH_SIZE" \
            --n-warmup "$WARMUP_STEPS" \
            --n-measure "$MEASURE_STEPS" \
            --forward-only
            
        # Forward + backward pass
        nsys profile $NSYS_OPTIONS \
            -o "nsys_profiles/${model_size}_seq${seq_len}_fwd_bwd" \
            python -m cs336_systems.nsys_benchmarking \
            --model-size "$model_size" \
            --seq-len "$seq_len" \
            --batch-size "$BATCH_SIZE" \
            --n-warmup "$WARMUP_STEPS" \
            --n-measure "$MEASURE_STEPS"
            
        # Complete training step
        nsys profile $NSYS_OPTIONS \
            -o "nsys_profiles/${model_size}_seq${seq_len}_full_training" \
            python -m cs336_systems.nsys_benchmarking \
            --model-size "$model_size" \
            --seq-len "$seq_len" \
            --batch-size "$BATCH_SIZE" \
            --n-warmup "$WARMUP_STEPS" \
            --n-measure "$MEASURE_STEPS" \
            --run-optimizer
        
        echo "Completed profiling for model size: $model_size, sequence length: $seq_len"
        echo ""
    done
done

echo "Profiling completed. Results are in the nsys_profiles directory."
echo "View results using NVIDIA Nsight Systems desktop application."
echo ""
echo "For each configuration, you'll find three profile files:"
echo "  - *_forward: Forward pass only (inference)"
echo "  - *_fwd_bwd: Forward + backward pass (training without optimizer)"
echo "  - *_full_training: Complete training step (forward + backward + optimizer)"
echo ""
echo "When analyzing the results, use the NVTX ranges to filter different phases:"
echo "  - warmup_phase: Ignore this phase in your analysis"
echo "  - measurement_phase: Focus on this phase for your analysis"
echo "  - forward_pass, backward_pass, optimizer_step: Individual phases"
echo "  - scaled_dot_product_attention: Details of attention operation"
echo "  - computing_attention_scores, computing_softmax, final_matmul: Sub-components of attention"