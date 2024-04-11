export DATASET_ROOT="/public/home/201810101923/datasets/opensora/dataset_v1.0.0_tmptest_sorted"
export DATASET_NAME="mixkit"

## Step1
#python scripts/panda70m/splitting/cutscene_detect.py  \
#  --root "${DATASET_ROOT}/${DATASET_NAME}"  \
#  --video-list "${DATASET_ROOT}/${DATASET_NAME}.txt"  \
#  --output-json-file "${DATASET_ROOT}/${DATASET_NAME}_cutscene_frame_idx.json"

## Step2
python scripts/panda70m/splitting/event_stitching.py  \
  --root "${DATASET_ROOT}/${DATASET_NAME}"  \
  --video-list "${DATASET_ROOT}/${DATASET_NAME}.txt"  \
  --cutscene-frameidx "${DATASET_ROOT}/${DATASET_NAME}_cutscene_frame_idx.json"  \
  --output-json-file "${DATASET_ROOT}/${DATASET_NAME}_event_timecode.json"

## Step3
python scripts/panda70m/splitting/video_splitting.py  \
  --root "${DATASET_ROOT}/${DATASET_NAME}"  \
  --video-list "${DATASET_ROOT}/${DATASET_NAME}.txt"  \
  --event-timecode "${DATASET_ROOT}/${DATASET_NAME}_event_timecode.json"  \
  --output-folder "${DATASET_ROOT}/${DATASET_NAME}_split"
