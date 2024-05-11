wandb offline

export DATA_PATH="/public/home/201810101923/datasets/opensora/dataset_v1.0.0_tmptest_sorted/sharegpt4v_path_cap_64x512x512.json"
export REPLACE_ROOT="/public/home/201810101923/datasets/opensora/dataset_v1.0.0_tmptest_sorted"
export MODEL_CACHE_DIR="/public/home/201810101923/models/opensora/v1.0.0"
export PRETRAINED_MODEL_PT="/public/home/201810101923/models/opensora/v1.0.0_sorted/opensora_stage3_65x512x512_bf16.pt"
export VIDEO_FOLDER="/remote-home1/dataset/data_split_tt"  # not used
python check_env/check_datasets.py \
  --model LatteT2V-XL/122 \
  --text_encoder_name DeepFloyd/t5-v1_1-xxl \
  --cache_dir ${MODEL_CACHE_DIR}  \
  --dataset t2v \
  --ae CausalVAEModel_4x8x8 \
  --ae_path CausalVAEModel_4x8x8 \
  --data_path ${DATA_PATH} \
  --replace_root ${REPLACE_ROOT}  \
  --video_folder ${VIDEO_FOLDER} \
  --sample_rate 1 \
  --num_frames 65 \
  --max_image_size 512 \
  --gradient_checkpointing \
  --attention_mode xformers \
  --train_batch_size=2 \
  --dataloader_num_workers 10 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=1000000 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="bf16" \
  --report_to="wandb" \
  --checkpointing_steps=500 \
  --output_dir="t2v-f65-512-img16-videovae488-bf16-ckpt-xformers-bs4-lr2e-5-t5" \
  --allow_tf32 \
  --pretrained t2v.pt \
  --use_deepspeed \
  --model_max_length 300 \
  --use_image_num 16 \
  --use_img_from_vid \
  --enable_tiling  \
  --enable_tracker  \
  --tracker_project_name check_debug \
  --tracker_run_name webvid_latent  \
  --logging_dir log_check_all
