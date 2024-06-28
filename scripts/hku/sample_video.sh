#!/bin/bash

set -x -e

# Set SLURM-like env params by hand
export SLURM_NNODES=1
export SLURM_PROCID=0

export NUM_GPUS=3

export OMP_NUM_THREADS=4

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))
export MASTER_ADDR="127.0.0.1"

RANK=${SLURM_PROCID}

echo "START TIME: $(date)"

export NCCL_NET=IB

# GPU nodes have no network
wandb offline


# please copy (or make a soft link by 'ln -s') the pretrained pipline to the current dir for more stable pipeline loading
# also remember to set "wandb offline" before training, and use syncwandb.sh to upload to wandb
export PYTHONPATH=${PWD}
export CACHE_DIR="/home/geyuan/pretrained/opensora/v1.0.0"
export PROMPT_LIST="examples/sora.txt"
export TRAIN_SIZE="v257x288x512"
export SAMPLE_SIZE="257x288x512"
export TRAIN_STEPS="58700"
export CKPT_PATH="/home/geyuan/pretrained/opensora/out_webvidlatent_${TRAIN_SIZE}_qknorm_rope/checkpoint-${TRAIN_STEPS}/model/diffusion_pytorch_model.safetensors"
export OUTPUT_DIR="./sample_videos/demo_webvidlatent${TRAIN_SIZE}_qknorm_rope_${TRAIN_STEPS}_${SAMPLE_SIZE}"
bash -c 'accelerate launch \
  --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
  --num_processes $(($NUM_GPUS * $SLURM_NNODES)) --num_machines $SLURM_NNODES --machine_rank $SLURM_PROCID \
  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  --gpu_ids "2,3,4" \
  opensora/sample/sample_t2v.py \
  --model_path LanguageBind/Open-Sora-Plan-v1.0.0 \
  --ckpt_path ${CKPT_PATH}  \
  --cache_dir ${CACHE_DIR} \
  --text_encoder_name DeepFloyd/t5-v1_1-xxl \
  --text_prompt ${PROMPT_LIST} \
  --ae CausalVAEModel_4x8x8 \
  --version ${TRAIN_SIZE} \
  --sample_size ${SAMPLE_SIZE} \
  --save_img_path ${OUTPUT_DIR} \
  --fps 24 \
  --guidance_scale 7.5 \
  --num_sampling_steps 50 \
  --enable_tiling  \
  --sample_method DDIM
  '

echo "DONE"