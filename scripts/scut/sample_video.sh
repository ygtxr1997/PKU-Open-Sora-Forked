#!/bin/bash
#SBATCH --job-name=opensora_sample
#SBATCH --partition=gpuA800
#SBATCH --nodes=1
#SBATCH --exclude=gpu[1]
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --gres=gpu:1               # number of gpus
#SBATCH --mem=500000MB                # memory
#SBATCH --output=outputs/%x-%j.out   # output file name
#SBATCH --time=30-00:00:00          # max time

set -x -e

# Set to equal gres=gpu:#
export NUM_GPUS=1

export OMP_NUM_THREADS=4

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

H=`hostname`
# RANK=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
RANK=${SLURM_PROCID}

# function to create the hostile
function makehostfile() {
       perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
       $slots=8 if $slots==0; # workaround 8 gpu machines
       @nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
       print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile

echo "START TIME: $(date)"

export NCCL_NET=IB

# GPU nodes have no network
wandb offline

# To save SLURM output logging files
if [ ! -d "outputs" ]; then
  mkdir -p "outputs"
fi

# Copy code from gitee to project directory
if [ `whoami` == "201810101923xxx" ]; then
  bash check_env/cp_code.sh
fi

# export MODEL_DIR="/exthome/future-technology-college-data/pretrained_models/pretrained_pipeline_fp16"
# please copy (or make a soft link by 'ln -s') the pretrained pipline to the current dir for more stable pipeline loading
# also remember to set "wandb offline" before training, and use syncwandb.sh to upload to wandb
export PYTHONPATH=${PWD}
export MODEL_DIR="pretrained_pipeline_fp16"
export PROMPT_LIST="examples/demo.txt"
export TRAIN_SIZE="17x288x512"
export TRAIN_STEPS="400"
export CKPT_PATH="/public/home/201810101923/code/PKU-Open-Sora-Forked/out_internvid_${TRAIN_SIZE}/checkpoint-${TRAIN_STEPS}/model/diffusion_pytorch_model.safetensors"
export OUTPUT_DIR="./sample_videos/demo_internvid${TRAIN_SIZE}_${TRAIN_STEPS}"
srun --jobid $SLURM_JOBID bash -c 'accelerate launch \
  --config_file check_env/check_deepspeed_config.yaml \
  --num_processes $(($NUM_GPUS * $SLURM_NNODES)) --num_machines $SLURM_NNODES --machine_rank $SLURM_PROCID \
  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  opensora/sample/sample_t2v.py \
  --model_path LanguageBind/Open-Sora-Plan-v1.0.0 \
  --ckpt_path ${CKPT_PATH}  \
  --cache_dir "/public/home/201810101923/models/opensora/v1.0.0" \
  --text_encoder_name DeepFloyd/t5-v1_1-xxl \
  --text_prompt ${PROMPT_LIST} \
  --ae CausalVAEModel_4x8x8 \
  --version ${TRAIN_SIZE} \
  --save_img_path ${OUTPUT_DIR} \
  --fps 24 \
  --guidance_scale 7.5 \
  --num_sampling_steps 250 \
  --enable_tiling
  '

echo "DONE"