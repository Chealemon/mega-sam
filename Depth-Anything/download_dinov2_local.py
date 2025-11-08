#!/usr/bin/env python3
"""
下载 DINOv2 模型到本地以避免网络问题
"""
import os
import sys
import torch

def download_dinov2_local():
    """下载 DINOv2 模型到本地目录"""
    
    # 设置下载目录
    local_dir = os.path.join(os.path.dirname(__file__), 'torchhub', 'facebookresearch_dinov2_main')
    os.makedirs(local_dir, exist_ok=True)
    
    models = ['vits', 'vitb', 'vitl']
    
    print("=" * 60)
    print("开始下载 DINOv2 模型到本地...")
    print("=" * 60)
    
    for model in models:
        print(f"\n正在下载 dinov2_{model}14...")
        try:
            # 尝试下载模型
            m = torch.hub.load(
                'facebookresearch/dinov2', 
                f'dinov2_{model}14',
                force_reload=False,
                trust_repo=True
            )
            print(f"✓ dinov2_{model}14 下载成功")
        except Exception as e:
            print(f"✗ dinov2_{model}14 下载失败: {e}")
            print(f"  请检查网络连接或使用代理")
    
    print("\n" + "=" * 60)
    print("下载完成！")
    print("现在 torch.hub 缓存中应该有模型了")
    print(f"缓存位置: {torch.hub.get_dir()}")
    print("\n如果下载成功，可以直接运行脚本（不需要 --localhub）")
    print("=" * 60)

if __name__ == '__main__':
    download_dinov2_local()
