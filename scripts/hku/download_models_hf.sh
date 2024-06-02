export CACHE_DIR="/home/geyuan/pretrained/opensora/v1.0.0"
export TARGETS="casual_vae"

# 1. Download from HuggingFace
python check_env/check_models.py --download  \
  --targets ${TARGETS} \
  --cache_dir ${CACHE_DIR}

# 2. Check local files
python check_env/check_models.py  \
  --targets ${TARGETS}  \
  --cache_dir ${CACHE_DIR}