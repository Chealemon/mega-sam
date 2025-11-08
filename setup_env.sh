#!/bin/bash
# MEGA-SAM 环境设置脚本
# 使用方法: source setup_env.sh

echo "========================================="
echo "设置 MEGA-SAM 运行环境"
echo "========================================="

# 激活conda环境
# 如果已经在conda环境中，尝试使用当前的conda
if command -v conda &> /dev/null; then
    conda activate mega_sam
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate mega_sam
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate mega_sam
else
    echo "✗ 找不到 conda，请确保已安装 Miniconda 或 Anaconda"
    return 1
fi

if [ $? -ne 0 ]; then
    echo "✗ 无法激活 mega_sam conda 环境"
    return 1
fi

echo "✓ Conda 环境已激活: mega_sam"

# 设置PyTorch库路径（自动检测）
CONDA_ENV_PATH=$(conda info --base)/envs/mega_sam
if [ -d "$CONDA_ENV_PATH/lib/python3.10/site-packages/torch/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"
    echo "✓ LD_LIBRARY_PATH 已设置"
fi

# 验证Python可用
python --version
echo ""

# 快速验证CUDA
python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "========================================="
echo "环境设置完成!"
echo "========================================="
echo ""
echo "可用命令:"
echo "  - 验证环境: cd base && python check_full_env.py"
echo "  - 重建DROID: cd base && bash rebuild_droid_slam.sh"
echo "  - 运行测试: cd camera_tracking_scripts && python test_sintel.py --datapath ../Sintel/alley_1"
echo "  - 深度估计: bash mono_depth_scripts/run_mono_depth.sh sintel"
echo "  - 评估模型: bash tools/evaluate.sh sintel"
echo ""
