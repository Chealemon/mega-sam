#!/usr/bin/env python3
"""
手动下载 UniDepth 权重文件的脚本
当自动下载失败时使用
"""
import os
import sys
import urllib.request
from pathlib import Path

def download_with_progress(url, dest_path):
    """下载文件并显示进度"""
    def reporthook(count, block_size, total_size):
        percent = min(int(count * block_size * 100 / total_size), 100)
        sys.stdout.write(f'\r下载进度: {percent}% [{count * block_size / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB]')
        sys.stdout.flush()
    
    print(f"正在从以下地址下载:\n{url}\n")
    urllib.request.urlretrieve(url, dest_path, reporthook)
    print("\n下载完成!")

def main():
    print("="*70)
    print("UniDepth V2 ViT-L14 权重文件手动下载工具")
    print("="*70)
    
    # 设置目标路径
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / "models--lpiccinelli--unidepth-v2-vitl14" / "snapshots" / "main"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    weight_file = model_dir / "pytorch_model.bin"
    
    print(f"\n目标路径: {weight_file}")
    
    if weight_file.exists():
        print(f"\n⚠ 文件已存在！")
        response = input("是否重新下载? (y/N): ")
        if response.lower() != 'y':
            print("取消下载")
            return
    
    # HuggingFace 直连地址
    url = "https://huggingface.co/lpiccinelli/unidepth-v2-vitl14/resolve/main/pytorch_model.bin"
    
    print("\n" + "="*70)
    print("开始下载 (文件大小约 1.5GB，请耐心等待)")
    print("="*70)
    print("\n提示:")
    print("1. 如果下载失败，请检查网络连接")
    print("2. 可以使用代理: export HTTP_PROXY=http://127.0.0.1:YOUR_PORT")
    print("3. 或者在浏览器中手动下载后放到上述目标路径\n")
    
    try:
        download_with_progress(url, weight_file)
        print(f"\n✓ 成功! 权重文件已保存到:\n  {weight_file}")
        print("\n现在可以运行 demo_mega-sam.py 了")
    except KeyboardInterrupt:
        print("\n\n下载已取消")
        if weight_file.exists():
            weight_file.unlink()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n手动下载步骤:")
        print("1. 在浏览器中访问:")
        print(f"   {url}")
        print("2. 下载文件后重命名为 pytorch_model.bin")
        print("3. 移动到:")
        print(f"   {weight_file}")
        sys.exit(1)

if __name__ == '__main__':
    main()
