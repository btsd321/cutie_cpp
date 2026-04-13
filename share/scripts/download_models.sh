#!/bin/bash
set -e

MODEL_DIR="$(dirname $(dirname "$0"))/model"
mkdir -p "$MODEL_DIR"

# 文件名和下载链接
MODELS=(
  "cutie-base-mega.pth https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth"
  "cutie-base-nomose.pth https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-nomose.pth"
  "cutie-base-wmose.pth https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-wmose.pth"
  "cutie-small-mega.pth https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-small-mega.pth"
  "cutie-small-nomose.pth https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-small-nomose.pth"
  "cutie-small-wmose.pth https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-small-wmose.pth"
)

for item in "${MODELS[@]}"; do
  set -- $item
  FILENAME=$1
  URL=$2
  DEST="$MODEL_DIR/$FILENAME"
  if [ -f "$DEST" ]; then
    echo "$FILENAME 已存在，跳过下载。"
  else
    echo "正在下载 $FILENAME ..."
    curl -L -o "$DEST" "$URL"
  fi
done

echo "所有模型下载完成。"
