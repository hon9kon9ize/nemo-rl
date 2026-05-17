#!/bin/bash -l
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -j

set -euo pipefail

module load cuda/12.9.1
module load gcc-toolset/12

eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
conda activate trl-vllm

PROJECT_DIR="${PROJECT_DIR:-/home/pj24001684/ku40000295/jc/projects/nemo-rl}"
cd "$PROJECT_DIR"

export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=ib0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_BLOCK_TIME_MS=7200000
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-nemo-rl-crossthink}"

MODEL="${MODEL:-/home/pj24001684/ku40000295/jc/projects/infinite-rl-gemma/outputs/dpo/merged_model}"
OUTPUT_DIR="${OUTPUT_DIR:-./nemo_rl_output/crossthink_yue_lora}"
GENERATION_LOG_FILE="${GENERATION_LOG_FILE:-$OUTPUT_DIR/generations.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
MAX_PROMPT="${MAX_PROMPT:-2048}"
MAX_COMPLETION="${MAX_COMPLETION:-1024}"

if [[ ! -f train.py ]]; then
  echo "train.py not found in PROJECT_DIR=$PROJECT_DIR" >&2
  exit 1
fi

if [[ ! -e "$MODEL" ]]; then
  echo "MODEL path does not exist: $MODEL" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python -u train.py \
  --model-id "$MODEL" \
  --dataset nemotron-crossthink \
  --dataset-split train \
  --max-samples "$MAX_SAMPLES" \
  --output-dir "$OUTPUT_DIR" \
  --generation-log-file "$GENERATION_LOG_FILE" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-generations 2 \
  --max-prompt-length "$MAX_PROMPT" \
  --max-completion-length "$MAX_COMPLETION" \
  --assistant-prefill-think \
  --normalize-prefilled-think \
  --reasoning-lang yue \
  --use-lora \
  --lora-rank 16 \
  --wandb
