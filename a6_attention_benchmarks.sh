#!/bin/bash

# Activate virtual environment if needed
# source venv/bin/activate

echo "============================================="
echo "CS336 Assignment 2 Attention Benchmarks"
echo "============================================="

# Part 1.2.1: PyTorch Attention Benchmark (uncompiled)
echo ""
echo "Part 1.2.1: Running uncompiled attention benchmark..."
python cs336_systems/pytorch_attention_benchmark.py
echo "Results saved to attention_benchmark_uncompiled.csv"

# Part 1.3: JIT-Compiled Attention Benchmark
echo ""
echo "Part 1.3: Running compiled attention benchmark..."
python cs336_systems/pytorch_attention_benchmark.py --compiled
echo "Results saved to attention_benchmark_compiled.csv"

echo ""
echo "Benchmarks complete! Results are in:"
echo "- attention_benchmark_uncompiled.csv (Part 1.2.1)"
echo "- attention_benchmark_compiled.csv (Part 1.3)" 