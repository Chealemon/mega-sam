#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时可视化 MegaSaM 运行过程
监控重建目录并实时显示最新结果
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm


def colorize_depth(depth, vmin=None, vmax=None):
    """将深度图转换为彩色可视化图像"""
    if vmin is None:
        vmin = np.percentile(depth, 5)
    if vmax is None:
        vmax = np.percentile(depth, 95)
    
    depth_normalized = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    
    cmap_func = cm.get_cmap('magma_r')
    colored = cmap_func(depth_normalized)
    colored_8bit = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return colored_8bit


def watch_reconstruction(scene_name, recon_dir, refresh_interval=2):
    """监控重建目录并实时显示"""
    print(f"\n{'='*60}")
    print(f"实时监控重建过程: {scene_name}")
    print(f"{'='*60}")
    print(f"监控目录: {recon_dir}/{scene_name}")
    print(f"刷新间隔: {refresh_interval} 秒")
    print(f"按 Ctrl+C 退出")
    print(f"{'='*60}\n")
    
    recon_path = Path(recon_dir) / scene_name
    
    last_frame_count = 0
    
    try:
        while True:
            # 检查文件是否存在
            images_file = recon_path / "images.npy"
            disps_file = recon_path / "disps.npy"
            
            if images_file.exists() and disps_file.exists():
                try:
                    # 加载数据
                    images = np.load(images_file, mmap_mode='r')
                    disps = np.load(disps_file, mmap_mode='r')
                    
                    current_frame_count = len(images)
                    
                    if current_frame_count > last_frame_count:
                        print(f"[{time.strftime('%H:%M:%S')}] 检测到新帧: {current_frame_count} 帧")
                        last_frame_count = current_frame_count
                        
                        # 显示最新一帧
                        idx = current_frame_count - 1
                        
                        # 图像
                        if images[idx].shape[0] == 3:  # CHW格式
                            img = images[idx].transpose(1, 2, 0)
                        else:
                            img = images[idx]
                        
                        # 深度
                        depth = 1.0 / (disps[idx] + 1e-8)
                        depth_colored = colorize_depth(depth)
                        
                        # 并排显示
                        img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                        depth_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
                        
                        # 调整大小使其一致
                        h, w = img_bgr.shape[:2]
                        depth_bgr = cv2.resize(depth_bgr, (w, h))
                        
                        # 拼接
                        combined = np.hstack([img_bgr, depth_bgr])
                        
                        # 添加文本
                        cv2.putText(combined, f"Frame {idx}/{current_frame_count}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(combined, "Image", (10, h-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(combined, "Depth", (w+10, h-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        # 显示
                        cv2.imshow(f'MegaSaM - {scene_name}', combined)
                        cv2.waitKey(1)
                        
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] 读取错误: {e}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 等待重建文件...")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"监控停止")
        print(f"总共处理: {last_frame_count} 帧")
        print(f"{'='*60}\n")
        cv2.destroyAllWindows()


def show_final_results(scene_name, recon_dir):
    """显示最终重建结果"""
    print(f"\n显示最终重建结果: {scene_name}")
    
    recon_path = Path(recon_dir) / scene_name
    
    if not recon_path.exists():
        print(f"错误: 重建目录不存在: {recon_path}")
        return
    
    try:
        images = np.load(recon_path / "images.npy")
        disps = np.load(recon_path / "disps.npy")
        poses = np.load(recon_path / "poses.npy")
        
        print(f"  Images: {images.shape}")
        print(f"  Disps: {disps.shape}")
        print(f"  Poses: {poses.shape}")
        
        # 交互式浏览
        current_idx = 0
        total_frames = len(images)
        
        print(f"\n使用方向键浏览帧:")
        print(f"  → / d: 下一帧")
        print(f"  ← / a: 上一帧")
        print(f"  Space: 播放/暂停")
        print(f"  ESC / q: 退出\n")
        
        playing = False
        
        while True:
            # 图像
            if images[current_idx].shape[0] == 3:
                img = images[current_idx].transpose(1, 2, 0)
            else:
                img = images[current_idx]
            
            # 深度
            depth = 1.0 / (disps[current_idx] + 1e-8)
            depth_colored = colorize_depth(depth)
            
            # 显示
            img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            depth_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
            
            h, w = img_bgr.shape[:2]
            depth_bgr = cv2.resize(depth_bgr, (w, h))
            combined = np.hstack([img_bgr, depth_bgr])
            
            # 添加信息
            info_text = f"Frame {current_idx+1}/{total_frames}"
            if playing:
                info_text += " [PLAYING]"
            cv2.putText(combined, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(f'MegaSaM Results - {scene_name}', combined)
            
            # 处理按键
            key = cv2.waitKey(33 if playing else 0) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or q
                break
            elif key == 83 or key == ord('d'):  # Right arrow or d
                current_idx = min(current_idx + 1, total_frames - 1)
            elif key == 81 or key == ord('a'):  # Left arrow or a
                current_idx = max(current_idx - 1, 0)
            elif key == 32:  # Space
                playing = not playing
            
            if playing:
                current_idx = (current_idx + 1) % total_frames
        
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description='实时可视化 MegaSaM 运行过程')
    parser.add_argument('--scene_name', type=str, required=True, help='场景名称')
    parser.add_argument('--recon_dir', type=str, default='reconstructions',
                        help='重建结果目录')
    parser.add_argument('--mode', type=str, default='watch',
                        choices=['watch', 'show'],
                        help='watch: 实时监控, show: 查看最终结果')
    parser.add_argument('--interval', type=int, default=2,
                        help='监控刷新间隔(秒)')
    
    args = parser.parse_args()
    
    if args.mode == 'watch':
        watch_reconstruction(args.scene_name, args.recon_dir, args.interval)
    else:
        show_final_results(args.scene_name, args.recon_dir)


if __name__ == '__main__':
    main()
