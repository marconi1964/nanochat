#!/bin/bash

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# Run as:
# bash dev/cpu_demo_run.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# This is also why I hide this script away in dev/

# extra set up for MacBook Pro MPS
export TORCH_DEVICE="mps"
export PYTORCH_ENABLE_MPS_FALLBACK=1

NO_DATASET=200 # 4 - 200 - 800
MAX_ITERS=100000
DEPTH=16
DEVICE_BATCH_SIZE=8
TOTAL_BATCH_SIZE=65536
# Check Claude Q&A
# The total_batch_size should be larger than world_tokens_per_fwdbwd and evenly divisible by it.
# You used --total_batch_size=1024 which is too small
# With device_batch_size=8 and max_seq_len=1024, 
# you need total_batch_size to be at least 8,192 and a multiple of it
MAX_CHARS=1000000000

# Fast Demo (depth=8, 6-12h): Quick testing
# Balanced (depth=12, 18-30h): Weekend project [RECOMMENDED for first try]
# High Quality (depth=16, 36-60h): Best quality/time balance
# Maximum (depth=18, 60-90h): Push your MacBook to its limits

# all the setup stuff
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# wipe the report
python -m nanochat.report reset

# train tokenizer on ~1B characters
python -m nanochat.dataset -n $NO_DATASET
python -m scripts.tok_train --max_chars=$MAX_CHARS
python -m scripts.tok_eval

# train a very small 4 layer model on the CPU
# each optimization step processes a single sequence of 1024 tokens
# we only run 50 steps of optimization (bump this to get better results)
python -m scripts.base_train \
    --depth=$DEPTH \
    --max_seq_len=1024 \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --eval_every=50 \
    --eval_tokens=4096 \
    --core_metric_every=50 \
    --core_metric_max_per_task=12 \
    --sample_every=50 \
    --num_iterations=$MAX_ITERS
python -m scripts.base_loss --device_batch_size=$DEVICE_BATCH_SIZE --split_tokens=4096
python -m scripts.base_eval --max-per-task=16

# midtraining
python -m scripts.mid_train \
    --max_seq_len=1024 \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --eval_every=50 \
    --eval_tokens=4096 \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --num_iterations=$MAX_ITERS
# eval results will be terrible, this is just to execute the code paths.
# note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20

# SFT
python -m scripts.chat_sft \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --target_examples_per_step=4 \
    --num_iterations=$MAX_ITERS \
    --eval_steps=4 \
    --eval_metrics_max_problems=16

# Chat CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
python -m scripts.chat_web

# python -m nanochat.report generate
