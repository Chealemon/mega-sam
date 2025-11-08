#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 MegaSaM 运行结果
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm


def colorize_depth(depth, vmin=None, vmax=None, cmap='magma_r'):
    """将深度图转换为彩色可视化图像"""
    if vmin is None:
        vmin = depth.min()
    if vmax is None:
        vmax = depth.max()
    
    # 归一化
    depth_normalized = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    
    # 应用colormap
    cmap_func = cm.get_cmap(cmap)
    colored = cmap_func(depth_normalized)
    
    # 转换为8bit RGB
    colored_8bit = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return colored_8bit


def visualize_depth_sequence(scene_name, depth_dir, output_dir, fps=10):
    """可视化深度序列"""
    print(f"\n=== 可视化深度序列: {scene_name} ===")
    
    # 查找深度文件
    depth_files = sorted(list(Path(depth_dir).glob('*.npy')))
    
    if len(depth_files) == 0:
        print(f"错误: 在 {depth_dir} 中没有找到 .npy 文件")
        return
    
    print(f"找到 {len(depth_files)} 个深度图")
    
    # 创建输出目录
    output_path = Path(output_dir) / scene_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 可视化每一帧
    for i, depth_file in enumerate(depth_files):
        depth = np.load(depth_file)
        
        # 彩色化
        colored = colorize_depth(depth)
        
        # 保存
        output_file = output_path / f"depth_{i:05d}.png"
        cv2.imwrite(str(output_file), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
        
        if i % 10 == 0:
            print(f"  处理进度: {i+1}/{len(depth_files)}")
    
    print(f"✓ 深度图已保存到: {output_path}")
    
    # 创建视频（如果有ffmpeg）
    try:
        video_file = output_path / f"{scene_name}_depth.mp4"
        cmd = f'ffmpeg -y -framerate {fps} -i "{output_path}/depth_%05d.png" -c:v libx264 -pix_fmt yuv420p "{video_file}"'
        os.system(cmd)
        print(f"✓ 视频已保存到: {video_file}")
    except:
        print("  (未安装ffmpeg，跳过视频生成)")


def visualize_reconstruction(scene_name, recon_dir, output_dir):
    """可视化重建结果"""
    print(f"\n=== 可视化重建结果: {scene_name} ===")
    
    recon_path = Path(recon_dir) / scene_name
    
    if not recon_path.exists():
        print(f"错误: 重建目录不存在: {recon_path}")
        return
    
    # 加载数据
    try:
        images = np.load(recon_path / "images.npy")
        disps = np.load(recon_path / "disps.npy")
        poses = np.load(recon_path / "poses.npy")
        intrinsics = np.load(recon_path / "intrinsics.npy")
        
        print(f"  Images shape: {images.shape}")
        print(f"  Disps shape: {disps.shape}")
        print(f"  Poses shape: {poses.shape}")
        print(f"  Intrinsics shape: {intrinsics.shape}")
    except Exception as e:
        print(f"错误: 无法加载重建数据: {e}")
        return
    
    # 创建输出目录
    output_path = Path(output_dir) / scene_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 可视化图像和深度
    num_frames = min(len(images), len(disps))
    
    for i in range(num_frames):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 原始图像
        if images[i].shape[0] == 3:  # CHW格式
            img = images[i].transpose(1, 2, 0)
        else:  # HWC格式
            img = images[i]
        
        axes[0].imshow(img)
        axes[0].set_title(f'Frame {i}')
        axes[0].axis('off')
        
        # 深度图
        depth = 1.0 / (disps[i] + 1e-8)
        im = axes[1].imshow(depth, cmap='magma_r')
        axes[1].set_title(f'Depth {i}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(output_path / f"frame_{i:05d}.png", dpi=100, bbox_inches='tight')
        plt.close()
        
        if i % 10 == 0:
            print(f"  处理进度: {i+1}/{num_frames}")
    
    print(f"✓ 重建可视化已保存到: {output_path}")
    
    # 可视化相机轨迹
    visualize_trajectory(poses, output_path / "trajectory.png")


def visualize_trajectory(poses, output_file):
    """可视化相机轨迹"""
    print(f"\n  生成相机轨迹图...")
    
    # 提取位置
    if poses.shape[-1] == 7:  # SE3 格式 [tx, ty, tz, qw, qx, qy, qz]
        positions = poses[:, :3]
    else:  # 4x4 矩阵
        positions = poses[:, :3, 3]
    
    # 绘制3D轨迹
    fig = plt.figure(figsize=(12, 10))
    
    # 3D视图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # XY平面
    ax2 = fig.add_subplot(222)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1)
    ax2.scatter(positions[0, 0], positions[0, 1], c='g', s=100, label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, label='End')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Plane')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # XZ平面
    ax3 = fig.add_subplot(223)
    ax3.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=1)
    ax3.scatter(positions[0, 0], positions[0, 2], c='g', s=100, label='Start')
    ax3.scatter(positions[-1, 0], positions[-1, 2], c='r', s=100, label='End')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Plane')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # YZ平面
    ax4 = fig.add_subplot(224)
    ax4.plot(positions[:, 1], positions[:, 2], 'b-', linewidth=1)
    ax4.scatter(positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax4.scatter(positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('YZ Plane')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 轨迹图已保存到: {output_file}")


def visualize_flow(scene_name, cache_flow_dir, output_dir):
    """可视化光流"""
    print(f"\n=== 可视化光流: {scene_name} ===")
    
    flow_path = Path(cache_flow_dir) / scene_name
    
    if not flow_path.exists():
        print(f"错误: 光流目录不存在: {flow_path}")
        return
    
    try:
        flows = np.load(flow_path / "flows.npy")
        flow_masks = np.load(flow_path / "flows_masks.npy")
        iijj = np.load(flow_path / "ii-jj.npy")
        
        print(f"  Flows shape: {flows.shape}")
        print(f"  Flow masks shape: {flow_masks.shape}")
        print(f"  ii-jj shape: {iijj.shape}")
        
        # 创建输出目录
        output_path = Path(output_dir) / scene_name / "flows"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 可视化部分光流
        num_vis = min(20, flows.shape[0])
        for i in range(num_vis):
            flow = flows[i].transpose(1, 2, 0)  # C,H,W -> H,W,C
            mask = flow_masks[i, 0]
            
            # 计算光流幅度
            flow_mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # 光流X分量
            im0 = axes[0].imshow(flow[:, :, 0], cmap='RdBu')
            axes[0].set_title(f'Flow X (frame {iijj[0, i]} -> {iijj[1, i]})')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0])
            
            # 光流Y分量
            im1 = axes[1].imshow(flow[:, :, 1], cmap='RdBu')
            axes[1].set_title(f'Flow Y')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1])
            
            # 光流幅度 + mask
            im2 = axes[2].imshow(flow_mag, cmap='hot')
            axes[2].contour(mask, levels=[0.5], colors='cyan', linewidths=2)
            axes[2].set_title(f'Flow Magnitude + Mask')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(output_path / f"flow_{i:03d}.png", dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"✓ 光流可视化已保存到: {output_path}")
        
    except Exception as e:
        print(f"错误: 无法加载光流数据: {e}")


def main():
    parser = argparse.ArgumentParser(description='可视化 MegaSaM 结果')
    parser.add_argument('--scene_name', type=str, required=True, help='场景名称')
    parser.add_argument('--depth_dir', type=str, default='Depth-Anything/video_visualization',
                        help='深度图目录')
    parser.add_argument('--recon_dir', type=str, default='reconstructions',
                        help='重建结果目录')
    parser.add_argument('--flow_dir', type=str, default='cache_flow',
                        help='光流缓存目录')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='输出目录')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'depth', 'reconstruction', 'flow'],
                        help='可视化模式')
    parser.add_argument('--fps', type=int, default=10, help='视频帧率')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"MegaSaM 结果可视化工具")
    print(f"{'='*60}")
    print(f"场景: {args.scene_name}")
    print(f"模式: {args.mode}")
    
    # 根据模式进行可视化
    if args.mode in ['all', 'depth']:
        depth_scene_dir = os.path.join(args.depth_dir, args.scene_name)
        if os.path.exists(depth_scene_dir):
            visualize_depth_sequence(args.scene_name, depth_scene_dir, 
                                    args.output_dir, args.fps)
        else:
            print(f"\n警告: 深度目录不存在: {depth_scene_dir}")
    
    if args.mode in ['all', 'reconstruction']:
        if os.path.exists(args.recon_dir):
            visualize_reconstruction(args.scene_name, args.recon_dir, args.output_dir)
        else:
            print(f"\n警告: 重建目录不存在: {args.recon_dir}")
    
    if args.mode in ['all', 'flow']:
        if os.path.exists(args.flow_dir):
            visualize_flow(args.scene_name, args.flow_dir, args.output_dir)
        else:
            print(f"\n警告: 光流目录不存在: {args.flow_dir}")
    
    print(f"\n{'='*60}")
    print(f"✓ 可视化完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
