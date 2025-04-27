uv venv
source .venv/bin/activate
uv pip install -e .
uv run python -m cs336_systems.benchmarking --model-size all --batch-size 4 --seq-len 128