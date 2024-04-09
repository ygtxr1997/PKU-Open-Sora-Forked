#!/bin/bash
SOURCE_GIT_DIR="/public/home/201810101923/code/gitee/PKU-Open-Sora-Gitee"
TARGET_PROJECT_DIR="/public/home/201810101923/code/PKU-Open-Sora-Forked"
BRANCH_NAME="yg_deployment"

echo "[1/2] Git cloning..."
cd ${SOURCE_GIT_DIR}
git pull
git checkout ${BRANCH_NAME}

echo "[2/2] Copying files..."
rsync -a --exclude='.git' ${SOURCE_GIT_DIR}/* ${TARGET_PROJECT_DIR}

echo "All files from ${SOURCE_GIT_DIR}:${BRANCH_NAME} copied to ${TARGET_PROJECT_DIR}"
