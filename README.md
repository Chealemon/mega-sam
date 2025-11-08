# MegaSaM

> **⚡ 快速开始**: `source setup_env.sh && bash quick_test.sh`  
> **🔧 RTX 5070 Ti 用户**: 已完全支持 sm_120 架构，开箱即用

[Project Page](https://mega-sam.github.io/index.html) | [Paper](https://arxiv.org/abs/2412.04463)

## 📋 目录

- [简介](#简介)
- [快速开始](#快速开始)
- [安装指南](#安装指南)
- [使用方法](#使用方法)
- [Shell 脚本参考](#shell-脚本参考)
- [常见问题](#常见问题)
- [可视化](#可视化)
- [引用](#引用)
- [许可证](#许可证)

---

## 📝 简介

**MegaSam: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos**

作者: Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, Noah Snavely

本代码库提供了 MegaSaM 的完整实现，用于从动态视频中进行准确、快速且鲁棒的结构与运动估计。

*This is not an officially supported Google product.*

---

## 🚀 快速开始

### 一键运行

```bash
# 克隆仓库（包含子模块）
git clone --recursive git@github.com:mega-sam/mega-sam.git
cd mega-sam

# 设置环境并测试
source setup_env.sh
bash quick_test.sh
```

### 项目结构

```
mega-sam/
├── setup_env.sh              # 环境设置脚本
├── quick_test.sh             # 快速测试脚本
├── base/                     # DROID-SLAM 核心
│   ├── rebuild_droid_slam.sh # 重新编译脚本
│   └── INSTALLATION.md       # 详细安装说明
├── camera_tracking_scripts/  # 相机跟踪脚本
├── mono_depth_scripts/       # 深度估计脚本
├── cvd_opt/                  # CVD 优化
├── tools/                    # 评估工具
├── Depth-Anything/           # 深度估计模块
├── UniDepth/                 # 统一深度估计
├── Sintel/                   # 数据集目录
└── checkpoints/              # 模型权重
```

---

## 📦 安装指南

### 环境要求

- Python 3.10
- CUDA 11.8+ (推荐 CUDA 12.x)
- PyTorch 2.0.1+
- 支持的 GPU: RTX 5070 Ti (sm_120), RTX 3090/4090, V100, A100 等

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate mega_sam
```

### 2. 安装 xformers (用于 UniDepth)

```bash
# 方式 1: 从预编译包安装（推荐）
wget https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
conda install xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2

# 方式 2: 从源码安装
# 参见 https://github.com/facebookresearch/xformers
```

### 3. 编译 DROID-SLAM 扩展

```bash
cd base
python setup.py install
```

**如果遇到问题**，使用自动重建脚本：

```bash
# 完整重建（首次安装或更换 GPU）
bash base/rebuild_droid_slam.sh --full

# 快速修复（仅更新 GPU 架构）
bash base/rebuild_droid_slam.sh --quick
```

### 4. 下载预训练权重

1. **DepthAnything**: 下载 [depth_anything_vitl14.pth](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth) 到 `Depth-Anything/checkpoints/`

2. **RAFT**: 下载 [raft-things.pth](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) 到 `cvd_opt/`

3. **MegaSaM**: 下载 megasam_final.pth 到 `checkpoints/`

### 5. 验证安装

```bash
source setup_env.sh
cd base && python check_full_env.py
```

应该看到：
```
✓ PyTorch 已安装
✓ CUDA 可用
✓ GPU 检测成功
✓ droid_backends 可以导入
✓ lietorch 可以导入
✓ droid 可以导入
✓ 所有检查通过！
```

---

## 🎯 使用方法

### Sintel 数据集

**1. 下载数据集**
```bash
# 下载并解压 Sintel 数据到 Sintel/ 目录
```

**2. 运行完整流程**
```bash
# 深度估计
bash mono_depth_scripts/run_mono_depth.sh sintel

# 相机跟踪（添加 --opt_focal 启用焦距优化）
bash tools/evaluate.sh sintel

# CVD 优化
bash cvd_opt/cvd_opt.sh sintel

# 评估
python evaluations_poses/evaluate_sintel.py
python evaluations_depth/evaluate_depth_ours_sintel.py
```

### DyCheck 数据集

```bash
# 下载 DyCheck 数据到 dycheck/ 目录

# 运行流程
bash mono_depth_scripts/run_mono_depth.sh dycheck
bash tools/evaluate.sh dycheck
bash cvd_opt/cvd_opt.sh dycheck

# 评估
python evaluations_poses/evaluate_dycheck.py
python evaluations_depth/evaluate_depth_ours_dycheck.py
```

### DAVIS 数据集（Demo）

```bash
# 下载 DAVIS 数据到 DAVIS/ 目录

# 运行流程
bash mono_depth_scripts/run_mono_depth.sh demo
bash tools/evaluate.sh demo
bash cvd_opt/cvd_opt.sh demo
```

---

## 🛠️ Shell 脚本参考

所有脚本已优化为相对路径，可从任意位置运行。

### 环境管理

```bash
# 设置环境（每次新会话必须运行）
source setup_env.sh

# 快速测试
bash quick_test.sh
```

### DROID-SLAM 编译

```bash
# 完整重建（首次安装/更换GPU）
bash base/rebuild_droid_slam.sh --full

# 快速修复（仅更新配置）
bash base/rebuild_droid_slam.sh --quick
```

### 深度估计

```bash
# 语法: bash mono_depth_scripts/run_mono_depth.sh [sintel|dycheck|demo]
bash mono_depth_scripts/run_mono_depth.sh sintel
```

自动运行：
- Depth-Anything (单目深度)
- UniDepth (度量深度)

### CVD 优化

```bash
# 语法: bash cvd_opt/cvd_opt.sh [sintel|dycheck|demo]
bash cvd_opt/cvd_opt.sh sintel
```

包含：
- RAFT 光流计算
- 一致性深度优化

### 模型评估

```bash
# 语法: bash tools/evaluate.sh [sintel|dycheck|demo] [--opt_focal]
bash tools/evaluate.sh sintel              # 基础评估
bash tools/evaluate.sh sintel --opt_focal  # 启用焦距优化
```

### DROID 数据集评估

```bash
# 语法: bash base/tools/evaluate_droid.sh [tartanair|tum|euroc|eth3d] [weights]
bash base/tools/evaluate_droid.sh tum
bash base/tools/evaluate_droid.sh euroc custom.pth
```

---

## ❓ 常见问题

### GPU 相关

**Q: 支持哪些 GPU？**

A: 支持以下架构：
- RTX 5070 Ti (sm_120) ✅
- RTX 4090/4080 (sm_89)
- RTX 3090/3080 (sm_86)
- A100 (sm_80)
- V100 (sm_70)
- 等等

**Q: 更换 GPU 后如何重新编译？**

```bash
cd base
bash rebuild_droid_slam.sh --full
```

### 编译问题

**Q: 出现 `undefined symbol` 错误？**

A: PyTorch 版本不匹配，重新编译：

```bash
cd base
pip uninstall droid_backends lietorch -y
bash rebuild_droid_slam.sh --full
```

**Q: CUDA 架构不匹配？**

A: 运行 GPU 检测并重新编译：

```bash
python base/check_gpu_arch.py
bash base/rebuild_droid_slam.sh --full
```

### 运行时问题

**Q: 找不到 `libc10.so` 或 `libtorch.so`？**

A: 设置库路径（已在 setup_env.sh 中自动设置）：

```bash
source setup_env.sh
```

**Q: 导入 droid 失败？**

A: 检查并验证安装：

```bash
cd base && python check_full_env.py
```

**Q: 网络问题无法下载模型？**

A: 参见 `Depth-Anything/NETWORK_FIX.md` 使用代理或离线下载。

### 数据集问题

**Q: 数据集路径如何配置？**

A: 脚本使用相对路径，数据集应放在项目根目录：
- Sintel → `./Sintel/`
- DyCheck → `./dycheck/`
- DAVIS → `./DAVIS/`

如果数据集在其他位置，修改脚本中的 `DATA_DIR` 变量。

---

## 🎨 可视化

### 查看结果

```bash
# 交互式查看
python visualize_live.py --scene_name mountain_1 --mode show

# 生成可视化
python visualize_results.py --scene_name mountain_1
```

### 实时监控

```bash
# 终端 1: 运行 MegaSaM
bash tools/evaluate.sh sintel

# 终端 2: 实时监控
python visualize_live.py --scene_name mountain_1 --mode watch
```

详细说明参见 `VISUALIZATION_GUIDE.md`。

---

## 📚 详细文档

- **`base/INSTALLATION.md`** - DROID-SLAM 详细安装说明
- **`base/DROID_COMPILE_FIX.md`** - 编译问题修复指南
- **`VISUALIZATION_GUIDE.md`** - 可视化使用指南
- **`Depth-Anything/NETWORK_FIX.md`** - 网络问题解决方案

---

## 📧 联系

如有关于论文的问题，请发邮件至: zl548@cornell.edu

---

## 📖 引用

```bibtex
@inproceedings{li2024_megasam,
  title     = {MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos},
  author    = {Li, Zhengqi and Tucker, Richard and Cole, Forrester and Wang, Qianqian and Jin, Linyi and Ye, Vickie and Kanazawa, Angjoo and Holynski, Aleksander and Snavely, Noah},
  booktitle = {arxiv},
  year      = {2024}
}
```

---

## 📄 许可证

**Copyright 2025 Google LLC**

软件部分采用 Apache License 2.0 授权。您可以在以下地址获取许可证副本：
https://www.apache.org/licenses/LICENSE-2.0

其他材料采用 Creative Commons Attribution 4.0 International License (CC-BY) 授权。您可以在以下地址获取许可证：
https://creativecommons.org/licenses/by/4.0/legalcode

除非适用法律要求或书面同意，否则根据 Apache 2.0 或 CC-BY 许可分发的所有软件和材料均按"原样"分发，不附带任何明示或暗示的保证或条件。详见许可证了解特定权限和限制。

*This is not an official Google product.*
