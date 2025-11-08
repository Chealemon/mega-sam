#!/bin/bash
# 快速验证和测试脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置环境
source "$SCRIPT_DIR/setup_env.sh"

# 验证安装
echo ""
echo "========================================="
echo "1. 验证 DROID-SLAM 安装"
echo "========================================="
cd "$SCRIPT_DIR/base"
python check_full_env.py

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ 环境验证失败,请运行: cd base && bash rebuild_droid_slam.sh"
    exit 1
fi

echo ""
read -p "环境验证通过! 是否运行Sintel测试? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "========================================="
    echo "2. 运行 Sintel 测试"
    echo "========================================="
    cd "$SCRIPT_DIR/camera_tracking_scripts"
    
    # 检查数据集是否存在
    if [ ! -d "$SCRIPT_DIR/Sintel/alley_1" ]; then
        echo "✗ 找不到 Sintel 数据集: Sintel/alley_1"
        echo "请确保数据集路径正确"
        exit 1
    fi
    
    # 运行测试
    python test_sintel.py --datapath "$SCRIPT_DIR/Sintel/alley_1"
    
    echo ""
    echo "========================================="
    echo "测试完成!"
    echo "========================================="
else
    echo "跳过测试"
fi
