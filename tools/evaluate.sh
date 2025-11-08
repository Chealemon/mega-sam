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

# 统一的评估脚本 - 支持 Sintel, DyCheck, Demo 三种数据集
# 使用方法: 
#   bash evaluate.sh sintel [--opt_focal]
#   bash evaluate.sh dycheck [--opt_focal]
#   bash evaluate.sh demo [--opt_focal]

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET=${1:-sintel}
CKPT_PATH="$PROJECT_ROOT/checkpoints/megasam_final.pth"

# 解析额外参数
shift
EXTRA_ARGS="$@"

# 根据数据集类型设置配置
case "$DATASET" in
  sintel)
    DATA_PATH="$PROJECT_ROOT/Sintel"
    TEST_SCRIPT="$PROJECT_ROOT/camera_tracking_scripts/test_sintel.py"
    evalset=(
      mountain_1 alley_1 alley_2 bamboo_1 bamboo_2
      temple_2 temple_3 market_2 market_5 market_6
      cave_4 ambush_4 ambush_5 ambush_6 cave_2
      shaman_3 sleeping_1 sleeping_2
    )
    ;;
  dycheck)
    DATA_PATH="$PROJECT_ROOT/dycheck"
    TEST_SCRIPT="$PROJECT_ROOT/camera_tracking_scripts/test_dycheck.py"
    evalset=(
      apple backpack block creeper handwavy haru-sit
      mochi-high-five pillow spin sriracha-tree teddy paper-windmill
    )
    ;;
  demo)
    DATA_PATH="$PROJECT_ROOT/DAVIS/JPEGImages/480p"
    TEST_SCRIPT="$PROJECT_ROOT/camera_tracking_scripts/test_demo.py"
    evalset=(swing breakdance-flare)
    ;;
  *)
    echo "错误: 未知的数据集类型: $DATASET"
    echo "用法: $0 [sintel|dycheck|demo] [--opt_focal]"
    exit 1
    ;;
esac

echo "========================================"
echo "运行评估: $DATASET"
echo "========================================"
echo "数据路径: $DATA_PATH"
echo "测试脚本: $TEST_SCRIPT"
echo "权重文件: $CKPT_PATH"
echo "场景列表: ${evalset[@]}"
echo "额外参数: ${EXTRA_ARGS:-无}"
echo ""

# 运行评估
for seq in ${evalset[@]}; do
  echo "----------------------------------------"
  echo "评估场景: $seq"
  echo "----------------------------------------"
  
  if [ "$DATASET" = "demo" ]; then
    # Demo 数据集使用特殊的路径结构
    CUDA_VISIBLE_DEVICES=0 python "$TEST_SCRIPT" \
      --datapath="$DATA_PATH/$seq" \
      --weights="$CKPT_PATH" \
      --scene_name $seq \
      --mono_depth_path "$PROJECT_ROOT/Depth-Anything/video_visualization" \
      --metric_depth_path "$PROJECT_ROOT/UniDepth/outputs" \
      --disable_vis $EXTRA_ARGS
  else
    # Sintel 和 DyCheck 使用标准路径
    CUDA_VISIBLE_DEVICES=0 python "$TEST_SCRIPT" \
      --datapath="$DATA_PATH" \
      --weights="$CKPT_PATH" \
      --scene_name $seq \
      --mono_depth_path "$PROJECT_ROOT/Depth-Anything/video_visualization" \
      --metric_depth_path "$PROJECT_ROOT/UniDepth/outputs" \
      --disable_vis $EXTRA_ARGS
  fi
  
  echo ""
done

echo "========================================"
echo "完成! 数据集: $DATASET"
echo "========================================"
