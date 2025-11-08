"""
临时补丁：强制禁用 xformers 避免 CUDA 错误
在运行 demo_mega-sam.py 之前先运行这个脚本
"""
import sys
import os

# 找到 attention.py 文件
unidepth_path = "/mnt/d/mega-sam/UniDepth/unidepth/models/backbones/metadinov2/attention.py"

print("="*60)
print("UniDepth xformers 禁用补丁")
print("="*60)

if not os.path.exists(unidepth_path):
    print(f"✗ 文件不存在: {unidepth_path}")
    sys.exit(1)

print(f"\n正在修改: {unidepth_path}")

# 读取文件
with open(unidepth_path, 'r') as f:
    content = f.read()

# 备份
backup_path = unidepth_path + ".backup"
if not os.path.exists(backup_path):
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✓ 已创建备份: {backup_path}")

# 强制设置 XFORMERS_AVAILABLE = False
if "XFORMERS_AVAILABLE = XFORMERS_AVAILABLE and torch.cuda.is_available()" in content:
    new_content = content.replace(
        "XFORMERS_AVAILABLE = XFORMERS_AVAILABLE and torch.cuda.is_available()",
        "XFORMERS_AVAILABLE = False  # Patched: Force disable xformers"
    )
    
    with open(unidepth_path, 'w') as f:
        f.write(new_content)
    
    print("✓ 已禁用 xformers")
    print("\n现在可以运行 demo_mega-sam.py 了")
    print("\n如需恢复，运行:")
    print(f"  cp {backup_path} {unidepth_path}")
else:
    print("✓ 已经是禁用状态或文件已修改")

print("="*60)
