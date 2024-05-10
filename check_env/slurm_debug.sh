#!/bin/bash
#SBATCH --job-name=multinode_debug
#SBATCH --partition=gpuA800
#SBATCH --nodes=4                   # number of nodes
#SBATCH --exclude=gpu[1]
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:8                # number of GPUs per node
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
export HOSTFILE="./hostfile"
function makehostfile() {
       perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
       $slots=8 if $slots==0; # workaround 8 gpu machines
       @nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
       print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > $HOSTFILE

export NCCL_NET=IB
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
######################

#export LAUNCHER="accelerate launch \
#    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml  \
#    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
#    --num_machines $SLURM_NNODES \
#    --main_process_ip ${MASTER_ADDR} \
#    --main_process_port ${MASTER_PORT} \
#    "
export LAUNCHER="torchrun \
  --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=$((SLURM_NNODES * GPUS_PER_NODE)) \
  --node_rank=\$SLURM_PROCID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT  \
  "
export PYTHONPATH=${PWD}
export DATA_PATH="/public/home/201810101923/datasets/opensora/dataset_v1.0.0_tmptest_sorted/sharegpt4v_path_cap_64x512x512.json"
export REPLACE_ROOT="/public/home/201810101923/datasets/opensora/dataset_v1.0.0_tmptest_sorted"
export MODEL_CACHE_DIR="/public/home/201810101923/models/opensora/v1.0.0"
export PRETRAINED_MODEL_PT="/public/home/201810101923/models/opensora/v1.0.0_sorted/panda70m_129x144x256/checkpoint-96000/model/diffusion_pytorch_model.safetensors"
export INTERNVID_DIR="/exthome/future-technology-college-data/Internvid_dataset/InternVid-10M-FLT-clip"
export INTERNVID_META="/exthome/future-technology-college-data/Internvid_dataset/InternVid-10M-flt-clips1.jsonl"
export PANDA70M_DIR="/public/home/201810101923/datasets/panda70m/clips_0"
export PANDA70M_META="/public/home/201810101923/datasets/panda70m/panda70m_training_clips_0.csv"
export WEBVID_DIR="/exthome/future-technology-college-data/202321063560/webvid_data/webvid_train_data"
export WEBVID_LATENT_DIR="/public/home/201810101923/datasets/webvid/latents"
export WEBVID_LATENT_META="/public/home/201810101923/datasets/webvid/latents_129x288x512/latents_meta_all.csv"
export OUTPUT_DIR="out_panda70m_129x288x512"
export VIDEO_FOLDER="/remote-home1/dataset/data_split_tt"  # not used
export SCRIPT="check_env/check_multi_nodes.py"
export SCRIPT_ARGS=" \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --cache_dir ${MODEL_CACHE_DIR}  \
    --dataset webvid_latent \
    --ae CausalVAEModel_4x8x8 \
    --ae_path CausalVAEModel_4x8x8 \
    --data_path ${DATA_PATH} \
    --replace_root ${REPLACE_ROOT}  \
    --video_folder ${VIDEO_FOLDER} \
    --sample_rate 1 \
    --num_frames 129 \
    --max_image_size 512 \
    --wh_ratio '16:9' \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size 2 \
    --dataloader_num_workers 6 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 1000000 \
    --learning_rate 2e-05 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --mixed_precision bf16 \
    --report_to wandb \
    --checkpointing_steps 500 \
    --output_dir ${OUTPUT_DIR} \
    --allow_tf32 \
    --pretrained ${PRETRAINED_MODEL_PT} \
    --use_deepspeed \
    --model_max_length 300 \
    --use_image_num 0 \
    --enable_tiling \
    --tracker_project_name scut_opensora \
    --tracker_run_name opensora512  \
    --resume_from_checkpoint latest  \
    --internvid_meta ${INTERNVID_META}  \
    --internvid_dir ${INTERNVID_DIR}  \
    --panda70m_meta ${PANDA70M_META}  \
    --panda70m_dir ${PANDA70M_DIR}  \
    --is_video_latent  \
    --webvid_meta ${WEBVID_LATENT_META}  \
    --webvid_dir ${WEBVID_LATENT_DIR}
    "

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
##srun --jobid $SLURM_JOBID bash -c "$CMD"
##srun torchrun \
#  --nnodes 4 \
#  --nproc_per_node 8 \
#  --rdzv_id $RANDOM \
#  --rdzv_backend c10d \
#  --rdzv_endpoint $MASTER_ADDR:29500 \
#  $SCRIPT $SCRIPT_ARGS
srun --jobid $SLURM_JOBID -n 4 bash -c 'deepspeed \
  --num_nodes 4 \
  --num_gpus 8 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --launcher SLURM \
  --hostfile=$HOSTFILE \
  $SCRIPT $SCRIPT_ARGS \
  --deepspeed
  '

##srun accelerate launch \
#  --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
#  --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
#  --num_machines $SLURM_NNODES \
#  --machine_rank $SLURM_PROCID \
#  --rdzv_backend c10d \
#  --rdzv_conf rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#  $SCRIPT $SCRIPT_ARGS
#All_ADDR=($(scontrol show hostnames $SLURM_JOB_NODELIST))
#for mrank in $(seq 0 $((SLURM_NNODES - 1)))
#do
#echo "$mrank address"=${All_ADDR[mrank]}
##srun --jobid $SLURM_JOBID -w ${All_ADDR[mrank]} bash -c "$LAUNCHER $SCRIPT $SCRIPT_ARGS" &
#done