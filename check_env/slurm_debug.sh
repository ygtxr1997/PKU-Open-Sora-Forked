#!/bin/bash
#SBATCH --job-name=multinode_debug
#SBATCH --partition=gpuA800
#SBATCH --nodes=4                   # number of nodes
#SBATCH --exclude=gpu[1]
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=64          # number of cores per tasks
#SBATCH --mem=500000MB              # memory
#SBATCH --output=outputs/%x-%j.out  # output file name
#SBATCH --time=30-00:00:00          # max time

######################
### Set enviroment ###
######################
set -x -e
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=4
######################

######################
#### Set network #####
######################
# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# function to create the hostile
function makehostfile() {
       perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
       $slots=8 if $slots==0; # workaround 8 gpu machines
       @nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
       print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile
export NCCL_NET=IB
######################

export LAUNCHER="accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml  \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    "
export SCRIPT="check_env/check_multi_nodes.py"
export SCRIPT_ARGS=' \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir ""  \
    --dataset webvid_latent \
    --ae CausalVAEModel_4x8x8 \
    --ae_path CausalVAEModel_4x8x8 \
    --data_path "" \
    --replace_root ""  \
    --video_folder "" \
    --sample_rate 1 \
    --num_frames 129 \
    --max_image_size 512 \
    --wh_ratio "16:9" \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=2 \
    --dataloader_num_workers 6 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=500 \
    --output_dir="" \
    --allow_tf32 \
    --pretrained "" \
    --use_deepspeed \
    --model_max_length 300 \
    --use_image_num 0 \
    --enable_tiling \
    --tracker_project_name scut_opensora \
    --tracker_run_name opensora512  \
    --resume_from_checkpoint "latest"  \
    '

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_SCRIPT_ARGS"
srun $CMD