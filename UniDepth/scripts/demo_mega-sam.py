import argparse
import glob
import os
import json

# ⚠️ 重要：在导入 torch 和其他库之前设置环境变量
# 这样可以在模块加载时就禁用 xformers
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['XFORMERS_DISABLED'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import cv2
import imageio
import numpy as np
from PIL import Image
import torch
import tqdm
from unidepth.models import UniDepthV2
from unidepth.utils import colorize, image_grid

# 强制禁用 xformers（在导入后再次确认）
try:
    import unidepth.models.backbones.metadinov2.attention as attn_module
    attn_module.XFORMERS_AVAILABLE = False
    print("✓ 已禁用 xformers")
except:
    pass

LONG_DIM = 640

def demo(model, args):
  outdir = args.outdir  # "./outputs"
  # os.makedirs(outdir, exist_ok=True)

  # for scene_name in scene_names:
  scene_name = args.scene_name
  outdir_scene = os.path.join(outdir, scene_name)
  os.makedirs(outdir_scene, exist_ok=True)
  # img_path_list = sorted(glob.glob("/home/zhengqili/filestore/DAVIS/DAVIS/JPEGImages/480p/%s/*.jpg"%scene_name))
  img_path_list = sorted(glob.glob(os.path.join(args.img_path, "*.jpg")))
  img_path_list += sorted(glob.glob(os.path.join(args.img_path, "*.png")))

  fovs = []
  for img_path in tqdm.tqdm(img_path_list):
    rgb = np.array(Image.open(img_path))[..., :3]
    if rgb.shape[1] > rgb.shape[0]:
      final_w, final_h = LONG_DIM, int(
          round(LONG_DIM * rgb.shape[0] / rgb.shape[1])
      )
    else:
      final_w, final_h = (
          int(round(LONG_DIM * rgb.shape[1] / rgb.shape[0])),
          LONG_DIM,
      )
    rgb = cv2.resize(
        rgb, (final_w, final_h), cv2.INTER_AREA
    )  # .transpose(2, 0, 1)

    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    # intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))
    # predict
    predictions = model.infer(rgb_torch)
    fov_ = np.rad2deg(
        2
        * np.arctan(
            predictions["depth"].shape[-1]
            / (2 * predictions["intrinsics"][0, 0, 0].cpu().numpy())
        )
    )
    depth = predictions["depth"][0, 0].cpu().numpy()
    print(fov_)
    fovs.append(fov_)
    # breakpoint()
    np.savez(
        os.path.join(outdir_scene, img_path.split("/")[-1][:-4] + ".npz"),
        depth=np.float32(depth),
        fov=fov_,
    )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--img-path", type=str)
  parser.add_argument("--outdir", type=str, default="./vis_depth")
  parser.add_argument("--scene-name", type=str)

  args = parser.parse_args()

  print("Torch version:", torch.__version__)
  print("CUDA available:", torch.cuda.is_available())
  if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
  
  # 使用本地配置和权重加载（完全离线，避免网络问题）
  print("\n" + "="*60)
  print("使用本地配置加载 UniDepth V2 模型")
  print("="*60)
  
  # 加载本地配置文件
  config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_v2_vitl14.json')
  config_path = os.path.abspath(config_path)
  print(f"\n1. 加载配置文件: {config_path}")
  
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"配置文件不存在: {config_path}")
  
  with open(config_path, 'r') as f:
    config = json.load(f)
  print("   ✓ 配置加载成功")
  
  # 创建模型
  print("\n2. 创建模型...")
  model = UniDepthV2(config)
  print("   ✓ 模型创建成功")
  
  # 加载权重
  print("\n3. 加载权重文件...")
  
  # 按优先级尝试多个可能的权重路径
  weight_paths = [
    # HuggingFace Hub 缓存路径
    os.path.expanduser("~/.cache/huggingface/hub/models--lpiccinelli--unidepth-v2-vitl14/snapshots/main/pytorch_model.bin"),
    # 备用：其他可能的 snapshot
    os.path.expanduser("~/.cache/huggingface/hub/models--lpiccinelli--unidepth-v2-vitl14/snapshots/*/pytorch_model.bin"),
    # Torch Hub 缓存路径（如果误放在这里）
    os.path.expanduser("~/.cache/torch/hub/models--lpiccinelli--unidepth-v2-vitl14/snapshots/main/pytorch_model.bin"),
  ]
  
  weight_path = None
  for path_pattern in weight_paths:
    import glob as glob_module
    matches = glob_module.glob(path_pattern)
    if matches:
      weight_path = matches[0]
      break
  
  if weight_path and os.path.exists(weight_path):
    print(f"   找到权重文件: {weight_path}")
    print(f"   文件大小: {os.path.getsize(weight_path) / (1024**3):.2f} GB")
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    print("   ✓ 权重加载成功")
  else:
    print("   ✗ 未找到权重文件")
    print("\n   请确保权重文件在以下位置之一:")
    for p in weight_paths:
      print(f"   - {p}")
    print("\n   下载方法:")
    print("   方式1: 运行下载脚本")
    print("     python UniDepth/scripts/download_weights.py")
    print("\n   方式2: 手动下载")
    print("     1. 访问: https://huggingface.co/lpiccinelli/unidepth-v2-vitl14/tree/main")
    print("     2. 下载: pytorch_model.bin")
    print("     3. 放到: ~/.cache/huggingface/hub/models--lpiccinelli--unidepth-v2-vitl14/snapshots/main/")
    raise FileNotFoundError("权重文件不存在")
  
  print("\n" + "="*60)
  print("模型加载完成，开始处理图像...")
  print("="*60 + "\n")
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"使用设备: {device}")
  
  if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
  
  model = model.to(device)
  model.eval()  # 设置为评估模式
  
  # 设置 resolution_level 避免警告
  if hasattr(model, 'resolution_level'):
    model.resolution_level = 0
  
  print("\n开始处理场景...")
  demo(model, args)
