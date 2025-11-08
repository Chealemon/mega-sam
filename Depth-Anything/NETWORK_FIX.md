# 解决 Depth-Anything 网络连接问题

## 问题原因
错误 `http.client.RemoteDisconnected` 表示无法连接到 GitHub 下载 DINOv2 模型。

## 解决方案

### 方案 1：使用代理（如果你在中国大陆）

在 WSL 终端中设置代理环境变量：

```bash
# 设置代理（根据你的实际代理地址修改）
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export NO_PROXY=localhost,127.0.0.1

# 然后运行脚本
cd /mnt/d/mega-sam/Depth-Anything
python run_videos.py --img-path <你的图片路径> --outdir ./vis_depth --encoder vitl --load-from ./checkpoints/depth_anything_vitl14.pth
```

### 方案 2：预先下载模型到缓存

运行下载脚本（需要有效的网络连接或代理）：

```bash
cd /mnt/d/mega-sam/Depth-Anything
python download_dinov2_local.py
```

这会将模型下载到 torch.hub 缓存，之后就可以离线使用。

### 方案 3：手动下载并使用本地模型

1. 在另一台能访问 GitHub 的机器上下载模型
2. 复制 `~/.cache/torch/hub/` 目录到当前机器
3. 使用 `--localhub` 参数运行

```bash
python run_videos.py --img-path <路径> --outdir ./vis_depth --encoder vitl --load-from <权重路径> --localhub
```

### 方案 4：修改 hosts 文件（临时方案）

如果 GitHub 访问不稳定，可以尝试修改 hosts：

```bash
sudo nano /etc/hosts
```

添加：
```
140.82.113.4 github.com
140.82.114.4 github.com
185.199.108.133 raw.githubusercontent.com
```

## 检查 torch.hub 缓存

查看已下载的模型：

```bash
ls -la ~/.cache/torch/hub/checkpoints/
ls -la ~/.cache/torch/hub/facebookresearch_dinov2_main/
```

## 推荐顺序

1. 先尝试方案 1（设置代理）
2. 如果代理可用，运行方案 2（预下载）
3. 之后就可以正常使用，不需要代理了
