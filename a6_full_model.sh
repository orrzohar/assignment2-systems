#!/bin/bash

# Activate virtual environment if needed
# source venv/bin/activate

echo "============================================="
echo "CS336 Assignment 2 Benchmarks"
echo "============================================="

# Part 1.2.1: PyTorch Attention Benchmark (uncompiled)
echo ""
echo "Part 1.2.1: Running uncompiled attention benchmark..."
python cs336_systems/pytorch_attention_benchmark.py --benchmark-type attention
echo "Results saved to attention_benchmark_uncompiled.csv"

# Part 1.3: JIT-Compiled Attention Benchmark
echo ""
echo "Part 1.3: Running compiled attention benchmark..."
python cs336_systems/pytorch_attention_benchmark.py --benchmark-type attention --compiled
echo "Results saved to attention_benchmark_compiled.csv"

# Part 1.3(b): Transformer Benchmark (uncompiled)
echo ""
echo "Part 1.3(b): Running uncompiled Transformer benchmark..."
python cs336_systems/nsys_benchmarking.py --model-size small --batch-size 4 --seq-len 128 --n-warmup 5 --n-measure 3 --run-optimizer
echo "Results saved to nsys profile output"

# Part 1.3(b): Transformer Benchmark (compiled)
echo ""
echo "Part 1.3(b): Running compiled Transformer benchmark..."
python cs336_systems/nsys_benchmarking.py --model-size small --batch-size 4 --seq-len 128 --n-warmup 5 --n-measure 3 --run-optimizer --compiled
echo "Results saved to nsys profile output"

echo ""
echo "============================================="
echo "All benchmarks complete!"
echo "============================================="
echo "Results are available in:"
echo "1. attention_benchmark_uncompiled.csv (Part 1.2.1)"
echo "2. attention_benchmark_compiled.csv (Part 1.3)"
echo "3. nsys profile output (Part 1.3(b))"
echo ""
echo "Use the LaTeX tables printed above for your report deliverables" 