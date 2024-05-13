#!/bin/bash
#SBATCH --job-name=extract_latents
#SBATCH --partition=gpuA800
#SBATCH --nodes=4
#SBATCH --exclude=gpu[1]
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --gres=gpu:8               # number of gpus
#SBATCH --mem=500000MB                # memory
#SBATCH --output=outputs/%x-%j.out   # output file name
#SBATCH --time=30-00:00:00          # max time

set -x -e

# Set to equal gres=gpu:#
export NUM_GPUS=8

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
export WANDB_DISABLE_SERVICE=True

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
export DATA_PATH="/public/home/201810101923/datasets/opensora/dataset_v1.0.0_tmptest_sorted/sharegpt4v_path_cap_64x512x512.json"
export REPLACE_ROOT="/public/home/201810101923/datasets/opensora/dataset_v1.0.0_tmptest_sorted"
export MODEL_CACHE_DIR="/public/home/201810101923/models/opensora/v1.0.0"
export PRETRAINED_MODEL_PT="/public/home/201810101923/models/opensora/v1.0.0_sorted/panda70m_129x144x256/checkpoint-96000/model/diffusion_pytorch_model.safetensors"
export INTERNVID_DIR="/exthome/future-technology-college-data/Internvid_dataset/InternVid-10M-FLT-clip"
export INTERNVID_META="/exthome/future-technology-college-data/Internvid_dataset/InternVid-10M-flt-clips1.jsonl"
export PANDA70M_DIR="/public/home/201810101923/datasets/panda70m/clips_0"
export PANDA70M_META="/public/home/201810101923/datasets/panda70m/panda70m_training_clips_0.csv"
export WEBVID_DIR="/exthome/future-technology-college-data/202321063560/webvid_data/webvid_train_data"
export OUTPUT_DIR="/public/home/201810101923/datasets/webavid/latents_v385x288x512/latents"
echo "num_gpus: $NUM_GPUS, slurm_nnodes: $SLURM_NNODES"
srun --jobid $SLURM_JOBID bash -c 'accelerate launch \
  --multi_gpu  \
  --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
  --num_processes $(($NUM_GPUS * $SLURM_NNODES)) --num_machines $SLURM_NNODES --machine_rank $SLURM_PROCID \
  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  opensora/dataset/save_features.py \
  --model LatteT2V-XL/122 \
  --text_encoder_name DeepFloyd/t5-v1_1-xxl \
  --cache_dir ${MODEL_CACHE_DIR}  \
  --dataset webvid \
  --ae CausalVAEModel_4x8x8 \
  --ae_path CausalVAEModel_4x8x8 \
  --data_path ${DATA_PATH} \
  --replace_root ${REPLACE_ROOT}  \
  --sample_rate 1 \
  --num_frames 385 \
  --use_smaller_frames  \
  --max_image_size 512 \
  --wh_ratio "16:9" \
  --extract_batch_size=2 \
  --dataloader_num_workers 6 \
  --max_extract_steps=1000000 \
  --mixed_precision="bf16" \
  --report_to="wandb" \
  --validation_steps=1000000 \
  --output_dir=${OUTPUT_DIR} \
  --allow_tf32 \
  --logging_dir="save_latents_log"  \
  --use_deepspeed \
  --model_max_length 300 \
  --use_image_num 0 \
  --enable_tiling  \
  --tile_overlap_factor 0.25  \
  --tracker_project_name scut_extract_latents \
  --tracker_run_name webvid  \
  --internvid_meta ${INTERNVID_META}  \
  --internvid_dir ${INTERNVID_DIR}  \
  --panda70m_meta ${PANDA70M_META}  \
  --panda70m_dir ${PANDA70M_DIR}  \
  --webvid_dir ${WEBVID_DIR}
  '

echo "DONE"