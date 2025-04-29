#!/bin/bash

# Create output directory
mkdir -p benchmark_results

# Run attention benchmark
echo "Running PyTorch attention benchmark..."
echo "Testing hidden dimensions: 1024, 2048, 4096, 8192"
echo "Testing sequence lengths: 1024, 4096, 8192, 16384"
echo "Using batch size: 8"

python cs336_systems/pytorch_attention_benchmark.py --benchmark-type attention

# Move attention results
mv attention_benchmark_results.csv benchmark_results/

# Run transformer benchmark
echo -e "\nRunning Transformer benchmark..."
echo "Testing model size: 2.7B"
echo "Testing sequence lengths: 128, 256, 512"

python cs336_systems/pytorch_attention_benchmark.py --benchmark-type transformer --model-size 2.7B

# Move transformer results
mv transformer_benchmark_results.csv benchmark_results/

echo -e "\nAll benchmarks completed. Results saved to benchmark_results/"
echo "You can find the LaTeX tables in the output above." 