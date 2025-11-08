# MegaSaM 可视化指南

本项目提供了多种可视化功能来帮助您理解和调试 MegaSaM 的运行过程。

## 📊 内置可视化功能

### 1. **DROID-SLAM 实时3D可视化**（需要 Open3D）

在 `base/demo.py` 和相机跟踪脚本中有实时3D点云可视化：

```python
# 在 base/droid_slam/visualization.py 中
# 使用 Open3D 实时显示相机轨迹和3D点云
```

**特点：**
- 实时显示相机位置和方向
- 3D点云重建
- 交互式视角控制

### 2. **深度图可视化**

Depth-Anything 会自动保存深度图：

```bash
# 深度图保存在
Depth-Anything/video_visualization/<scene_name>/*.npy
```

### 3. **光流可视化**

使用 `cvd_opt/core/utils/flow_viz.py` 中的工具：

```python
from cvd_opt.core.utils.flow_viz import flow_to_image

# 将光流转换为彩色可视化图像
flow_img = flow_to_image(flow_uv)
```

## 🎨 新增可视化工具

我为您创建了两个新的可视化脚本：

### 1. **visualize_results.py** - 完整结果可视化

生成高质量的可视化图像和视频。

**用法：**

```bash
# 可视化所有内容（深度、重建、光流）
python visualize_results.py --scene_name mountain_1

# 只可视化深度序列
python visualize_results.py --scene_name mountain_1 --mode depth

# 只可视化重建结果
python visualize_results.py --scene_name mountain_1 --mode reconstruction

# 只可视化光流
python visualize_results.py --scene_name mountain_1 --mode flow

# 自定义输出目录和帧率
python visualize_results.py --scene_name mountain_1 --output_dir my_visualizations --fps 30
```

**输出内容：**
- 彩色深度图序列
- 深度视频（需要 ffmpeg）
- 图像+深度对比图
- 相机轨迹图（3D + 多个2D平面）
- 光流可视化

### 2. **visualize_live.py** - 实时监控

在 MegaSaM 运行时实时查看结果。

**用法：**

```bash
# 实时监控重建过程（在另一个终端运行）
python visualize_live.py --scene_name mountain_1 --mode watch --interval 2

# 查看最终结果（交互式浏览）
python visualize_live.py --scene_name mountain_1 --mode show
```

**交互控制（show模式）：**
- `→` 或 `d`: 下一帧
- `←` 或 `a`: 上一帧
- `Space`: 播放/暂停
- `ESC` 或 `q`: 退出

## 📁 可视化输出目录结构

运行可视化后，会生成以下结构：

```
visualizations/
├── mountain_1/
│   ├── depth_00000.png          # 彩色深度图
│   ├── depth_00001.png
│   ├── ...
│   ├── mountain_1_depth.mp4     # 深度视频
│   ├── frame_00000.png          # 图像+深度对比
│   ├── ...
│   ├── trajectory.png           # 相机轨迹
│   └── flows/
│       ├── flow_000.png         # 光流可视化
│       └── ...
```

## 🔧 完整工作流程示例

### 方案1: 运行后可视化

```bash
# 1. 运行 MegaSaM 流程
cd /mnt/d/mega-sam
bash camera_tracking_scripts/test_sintel.py --scene_name mountain_1

# 2. 生成可视化
python visualize_results.py --scene_name mountain_1

# 3. 交互式查看结果
python visualize_live.py --scene_name mountain_1 --mode show
```

### 方案2: 实时监控

```bash
# 终端1: 运行 MegaSaM
cd /mnt/d/mega-sam
bash camera_tracking_scripts/test_sintel.py --scene_name mountain_1

# 终端2: 实时监控（同时运行）
python visualize_live.py --scene_name mountain_1 --mode watch
```

## 📊 可视化示例

### 深度图可视化
- 使用 `magma_r` colormap（暖色=近，冷色=远）
- 自动调整动态范围（5th-95th percentile）

### 轨迹可视化
- 3D轨迹图
- XY平面（俯视图）
- XZ平面（侧视图）
- YZ平面（前视图）
- 绿点=起点，红点=终点

### 光流可视化
- X分量（红蓝色图）
- Y分量（红蓝色图）
- 幅度（热力图）
- 青色轮廓=有效区域mask

## 🔍 调试技巧

### 1. 检查重建是否正常

```python
import numpy as np

# 加载数据
images = np.load('reconstructions/mountain_1/images.npy')
disps = np.load('reconstructions/mountain_1/disps.npy')
poses = np.load('reconstructions/mountain_1/poses.npy')

print(f"帧数: {len(images)}")
print(f"图像范围: [{images.min()}, {images.max()}]")
print(f"视差范围: [{disps.min()}, {disps.max()}]")
```

### 2. 查看深度统计

```python
depth = 1.0 / (disps + 1e-8)
print(f"深度范围: [{depth.min():.2f}, {depth.max():.2f}] 米")
print(f"中位深度: {np.median(depth):.2f} 米")
```

### 3. 检查相机移动

```python
# 计算相机移动距离
if poses.shape[-1] == 7:  # SE3格式
    positions = poses[:, :3]
else:  # 4x4矩阵
    positions = poses[:, :3, 3]

distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
print(f"平均帧间移动: {distances.mean():.4f}")
print(f"总移动距离: {distances.sum():.4f}")
```

## 📦 依赖项

可视化脚本需要以下Python包：

```bash
pip install numpy opencv-python matplotlib
```

可选（用于生成视频）：
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# 或者使用conda
conda install ffmpeg
```

## 🎯 快速开始

最简单的使用方式：

```bash
# 1. 确保已经运行了重建
ls reconstructions/mountain_1/

# 2. 交互式查看结果
python visualize_live.py --scene_name mountain_1 --mode show

# 3. 生成所有可视化
python visualize_results.py --scene_name mountain_1
```

## 💡 提示

1. **实时监控**：在运行耗时较长的场景时，使用 `visualize_live.py --mode watch` 可以及时发现问题
2. **视频生成**：生成的MP4视频可以方便地分享和演示结果
3. **轨迹分析**：轨迹图可以帮助判断相机定位是否准确
4. **光流检查**：光流可视化可以帮助诊断动态场景和遮挡问题

## 🐛 常见问题

**Q: OpenCV窗口没有显示？**
A: 在WSL中需要安装X server（如VcXsrv），或者只使用保存图片的功能。

**Q: 内存不足？**
A: 使用 `mmap_mode='r'` 加载大文件，或者减少可视化的帧数。

**Q: 视频生成失败？**
A: 检查是否安装了ffmpeg，或者只生成图片序列。

---

享受可视化！ 🎨✨
