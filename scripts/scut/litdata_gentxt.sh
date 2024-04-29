export INPUT_DATA_FOLDER="/public/home/201810101923/datasets/panda70m/clips_0/"
export INPUT_META_PATH="/public/home/201810101923/datasets/panda70m/panda70m_training_full.csv"
export SAVE_FOLDER="/public/home/201810101923/datasets/panda70m/"
export MAX_LEN="-1"

python opensora/dataset/litdata_process.py  \
      -m gen_txt  \
      --input_data_folder ${INPUT_DATA_FOLDER}  \
      --input_meta_path ${INPUT_META_PATH}  \
      --save_folder ${SAVE_FOLDER}  \
      --max_len ${MAX_LEN}
