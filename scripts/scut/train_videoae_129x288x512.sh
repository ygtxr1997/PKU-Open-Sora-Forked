#!/bin/bash
#SBATCH --job-name=opensora_train
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
export PRETRAINED_MODEL_PT="/public/home/201810101923/models/opensora/v1.0.0_sorted/out_panda70m_129x288x512/checkpoint-36000/model/diffusion_pytorch_model.safetensors"
export INTERNVID_DIR="/exthome/future-technology-college-data/Internvid_dataset/InternVid-10M-FLT-clip"
export INTERNVID_META="/exthome/future-technology-college-data/Internvid_dataset/InternVid-10M-flt-clips1.jsonl"
export PANDA70M_DIR="/public/home/201810101923/datasets/panda70m/clips_0"
export PANDA70M_META="/public/home/201810101923/datasets/panda70m/panda70m_training_clips_0.csv"
export WEBVID_DIR="/exthome/future-technology-college-data/202321063560/webvid_data/webvid_train_data"
export WEBVID_LATENT_DIR="/public/home/201810101923/datasets/webvid/latents_129x288x512/latents"
export WEBVID_LATENT_META="/public/home/201810101923/datasets/webvid/latents_129x288x512/latents_meta_all.csv"
export OUTPUT_DIR="out_webvidlatent_129x288x512"
export VIDEO_FOLDER="/remote-home1/dataset/data_split_tt"  # not used
srun --jobid $SLURM_JOBID bash -c 'accelerate launch \
  --multi_gpu \
  --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
  --num_processes $(($NUM_GPUS * $SLURM_NNODES)) --num_machines $SLURM_NNODES --machine_rank $SLURM_PROCID \
  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  opensora/train/train_t2v.py \
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
  --output_dir ${OUTPUT_DIR} \
  --allow_tf32 \
  --pretrained ${PRETRAINED_MODEL_PT} \
  --use_deepspeed \
  --model_max_length 300 \
  --use_image_num 0 \
  --enable_tiling \
  --tracker_project_name scut_opensora \
  --tracker_run_name opensora512  \
  --resume_from_checkpoint "latest"  \
  --internvid_meta ${INTERNVID_META}  \
  --internvid_dir ${INTERNVID_DIR}  \
  --panda70m_meta ${PANDA70M_META}  \
  --panda70m_dir ${PANDA70M_DIR}  \
  --is_video_latent  \
  --webvid_meta ${WEBVID_LATENT_META}  \
  --webvid_dir ${WEBVID_LATENT_DIR}
  '

echo "DONE"