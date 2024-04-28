export INPUT_DATA_FOLDER="/exthome/future-technology-college-data/date-disk-0426/disk-2/disk-2/Panda-70M/litdata_0"
export SAVE_FOLDER="/public/home/201810101923/datasets/panda70m/clips_0"
export MAX_LEN="-1"

python opensora/dataset/litdata_process.py  \
  --input_data_folder ${INPUT_DATA_FOLDER}  \
  --save_folder ${SAVE_FOLDER}  \
  --max_len ${MAX_LEN}