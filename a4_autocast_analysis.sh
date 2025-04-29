#!/bin/bash

# Script to run mixed precision benchmarking experiments
echo "=============================================================="
echo "Casting data types: float16"
python cs336_systems/mixed_precision_toy_model.py
echo "============================="

echo "Casting data types: bfloat16"
python cs336_systems/mixed_precision_toy_model.py --precision bfloat16
echo "=============================================================="

