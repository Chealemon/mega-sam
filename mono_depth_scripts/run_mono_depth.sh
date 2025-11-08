#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 统一的深度估计脚本 - 支持 Sintel, DyCheck, Demo 三种数据集
# 使用方法: 
#   bash run_mono_depth.sh sintel
#   bash run_mono_depth.sh dycheck
#   bash run_mono_depth.sh demo

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET=${1:-sintel}

# 根据数据集类型设置配置
case "$DATASET" in
  sintel)
    DATA_DIR="$PROJECT_ROOT/Sintel"
    IMG_SUBPATH="rgb"
    evalset=(
      mountain_1 alley_1 alley_2 bamboo_1 bamboo_2
      temple_2 temple_3 market_2 market_5 market_6
      cave_4 ambush_4 ambush_5 ambush_6 cave_2
      shaman_3 sleeping_1 sleeping_2
    )
    ;;
  dycheck)
    DATA_DIR="$PROJECT_ROOT/dycheck"
    IMG_SUBPATH="dense/images"
    evalset=(
      apple backpack block creeper handwavy haru-sit
      mochi-high-five pillow spin sriracha-tree teddy paper-windmill
    )
    ;;
  demo)
    DATA_DIR="$PROJECT_ROOT/DAVIS/JPEGImages/480p"
    IMG_SUBPATH=""
    evalset=(swing breakdance-flare)
    ;;
  *)
    echo "错误: 未知的数据集类型: $DATASET"
    echo "用法: $0 [sintel|dycheck|demo]"
    exit 1
    ;;
esac

echo "========================================"
echo "运行深度估计: $DATASET"
echo "========================================"
echo "数据路径: $DATA_DIR"
echo "场景列表: ${evalset[@]}"
echo ""

# Run DepthAnything
echo "----------------------------------------"
echo "1. 运行 Depth-Anything..."
echo "----------------------------------------"
for seq in ${evalset[@]}; do
  if [ -n "$IMG_SUBPATH" ]; then
    IMG_PATH="$DATA_DIR/$seq/$IMG_SUBPATH"
  else
    IMG_PATH="$DATA_DIR/$seq"
  fi
  
  echo "处理场景: $seq"
  CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/Depth-Anything/run_videos.py" --encoder vitl \
    --load-from "$PROJECT_ROOT/Depth-Anything/checkpoints/depth_anything_vitl14.pth" \
    --img-path "$IMG_PATH" \
    --outdir "$PROJECT_ROOT/Depth-Anything/video_visualization/$seq"
done

echo ""
echo "----------------------------------------"
echo "2. 运行 UniDepth..."
echo "----------------------------------------"

# Run UniDepth
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT/UniDepth"

for seq in ${evalset[@]}; do
  if [ -n "$IMG_SUBPATH" ]; then
    IMG_PATH="$DATA_DIR/$seq/$IMG_SUBPATH"
  else
    IMG_PATH="$DATA_DIR/$seq"
  fi
  
  echo "处理场景: $seq"
  CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/UniDepth/scripts/demo_mega-sam.py" \
    --scene-name $seq \
    --img-path "$IMG_PATH" \
    --outdir "$PROJECT_ROOT/UniDepth/outputs"
done

echo ""
echo "========================================"
echo "完成! 数据集: $DATASET"
echo "========================================"
