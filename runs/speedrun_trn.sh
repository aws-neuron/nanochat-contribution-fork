#!/bin/bash
set -euo pipefail
# Name the stage that failed: with `set -e` a failing background stage (e.g. the CPU sample eval) otherwise aborts the
# run at the `wait` with no hint of which command it was.
trap 'echo "speedrun failed at line $LINENO: $BASH_COMMAND" >&2' ERR

# nanochat Speedrun on AWS Trainium (64 LNC2 NeuronCores). The Trainium counterpart of runs/speedrun.sh:
# same stages, same order; it differs only in the stage commands (rank counts, the Trainium recipe flags, LNC1 for
# chat_eval), the fresh-cache note, the compile-service guard and the base-dir guard.
#
# 1) Example launch (simplest):
# bash runs/speedrun_trn.sh
# 2) Example launch in a screen session (because the run takes ~2 hours):
# screen -L -Logfile runs/speedrun_trn.log -S speedrun bash runs/speedrun_trn.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun_trn.log -S speedrun bash runs/speedrun_trn.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat (base-dir guard: an exported NANOCHAT_BASE_DIR
# wins, so several runs on one host keep their checkpoints apart)
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv: the Neuron SDK venv must already be active (neuronx-cc must resolve on PATH; torch_neuronx silently
# deadlocks in synchronize() otherwise).
command -v neuronx-cc >/dev/null || { echo "neuronx-cc not on PATH: activate the Neuron venv first"; exit 1; }
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_CC_FLAGS="--optlevel 1"
export TORCH_NEURONX_LOG_LEVEL=WARNING

# Compile-service guard. torch_neuronx compiles NEFFs through an asynchronous worker pool unless
# NEURON_COMPILE_SERVICE=0 is exported (some shells do that so a SIGINT cannot kill a shared service). A
# speedrun launched from such a shell would compile every stage synchronously: +100 s on cold step 0, base_eval
# 7.5 -> 9.5 min, chat_eval 16 -> 24 min. Drop an inherited value.
if [ -n "${NEURON_COMPILE_SERVICE:-}" ]; then
    echo "note: dropping inherited NEURON_COMPILE_SERVICE=$NEURON_COMPILE_SERVICE (the async compile service stays on)"
    unset NEURON_COMPILE_SERVICE
fi

# Fresh cache: torch_neuronx defaults TORCH_NEURONX_HLO_CACHE_DIR to /tmp/hlo_cache, so a speedrun pays a cold
# compile (the headline number includes it). Export TORCH_NEURONX_HLO_CACHE_DIR / TORCH_NEURONX_NEFF_CACHE_DIR to
# persistent directories yourself for iterative dev runs. Within one speedrun the SFT stage reuses the pretrain
# Block graphs from the same cache.
echo "Compile caches: HLO=${TORCH_NEURONX_HLO_CACHE_DIR:-/tmp/hlo_cache (default)} NEFF=${TORCH_NEURONX_NEFF_CACHE_DIR:-<torch_neuronx default>}"

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "${WANDB_RUN:-}" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
# The maximum total number of shards available in the entire dataset is 6542.
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data (skipped when it already exists)
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    python -m scripts.tok_train
    # evaluate the tokenizer (report compression ratio etc.)
    python -m scripts.tok_eval
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
# Trainium recipe: 64 LNC2 ranks, MBS 8 (grad_accum 1 at 8x2048x64 = 1,048,576 tokens), FP8 on the MLP Linears,
# activation checkpointing on every Block (~10 GiB per rank; SFT inherits it from the checkpoint meta), no in-loop
# eval (a second MBS=8 dataloader + forward does not fit next to the training buffers; base_eval below measures
# CORE+bpb on the final checkpoint), 2304 MiB collective buckets with split reduce-scatter / all-gather.
NEURON_USE_SPLIT_REDUCE_SCATTER=1 NEURON_USE_SPLIT_ALLGATHER=1 COLLECTIVE_BUCKETSIZE_IN_MB=2304 \
torchrun --standalone --nproc_per_node=64 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=8 --activation-checkpoint=1 --fp8 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val (64 ranks, fixed-shape CORE batches), and draw samples (CPU:
# generative decode of short prompts is faster there and uses no NeuronCore, so it runs alongside)
OMP_NUM_THREADS=16 python -m scripts.base_eval --eval sample --device-type=cpu &
BASE_EVAL_SAMPLE_PID=$!
torchrun --standalone --nproc_per_node=64 -m scripts.base_eval -- --eval core,bpb --device-batch-size=4
wait $BASE_EVAL_SAMPLE_PID

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# run SFT and eval the model. SFT: the pretrain step (MBS 8, FP8 MLP, activation checkpointing, same Block graphs -> compile
# cache hits; activation checkpointing comes from the base checkpoint's GPTConfig); the SFT optimizer shards have no
# consumer, so they are not written (64 x 6 GB). The datasets are prefetched once in one process, so the 64 / 128
# ranks start from the local cache (tasks/common.py load_hub_dataset).
python -c "
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.arc import ARC
from tasks.humaneval import HumanEval
SmolTalk(split='train'); SmolTalk(split='test')
MMLU(subset='all', split='auxiliary_train'); MMLU(subset='all', split='test')
GSM8K(subset='main', split='train'); GSM8K(subset='main', split='test')
ARC(subset='ARC-Easy', split='test'); ARC(subset='ARC-Challenge', split='test')
HumanEval()
print('SFT / chat_eval datasets cached')
"
NEURON_USE_SPLIT_REDUCE_SCATTER=1 NEURON_USE_SPLIT_ALLGATHER=1 COLLECTIVE_BUCKETSIZE_IN_MB=2304 \
torchrun --standalone --nproc_per_node=64 -m scripts.chat_sft -- --device-batch-size=8 --fp8 --eval-every=-1 --chatcore-every=-1 --save-optimizer=0 --run=$WANDB_RUN
# chat_eval on 128 LNC1 ranks (one physical core each: ~2x the generative throughput of LNC2-64), static-shape
# kv-cache decode with prefill buckets and 16 problems decoded per rank concurrently (nanochat/trn/engine.py)
NEURON_LOGICAL_NC_CONFIG=1 \
torchrun --standalone --nproc_per_node=128 -m scripts.chat_eval -- -i sft --device-type=neuron

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"
