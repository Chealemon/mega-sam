# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Preprocess flow for MegaSaM."""

import glob
import os
import sys

# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top

import numpy as np
import torch
# FLOW ESTIMATOR
sys.path.append('cvd_opt/core')
from raft import RAFT
from core.utils.utils import InputPadder
from pathlib import Path  # pylint: disable=g-importing-member

import argparse
import tqdm
import cv2


def warp_flow(img, flow):
  h, w = flow.shape[:2]
  flow_new = flow.copy()
  flow_new[:, :, 0] += np.arange(w)
  flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

  res = cv2.remap(
      img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
  )
  return res


def resize_flow(flow, img_h, img_w):
  # flow = np.load(flow_path)
  flow_h, flow_w = flow.shape[0], flow.shape[1]
  flow[:, :, 0] *= float(img_w) / float(flow_w)
  flow[:, :, 1] *= float(img_h) / float(flow_h)
  flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)

  return flow


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', default='raft-things.pth', help='restore checkpoint'
  )
  parser.add_argument('--small', action='store_true', help='use small model')
  parser.add_argument('--scene_name', type=str, help='use small model')
  parser.add_argument('--datapath')

  parser.add_argument('--path', help='dataset for evaluation')
  parser.add_argument(
      '--num_heads',
      default=1,
      type=int,
      help='number of heads in attention and aggregation',
  )
  parser.add_argument(
      '--position_only',
      default=False,
      action='store_true',
      help='only use position-wise attention',
  )
  parser.add_argument(
      '--position_and_content',
      default=False,
      action='store_true',
      help='use position and content-wise attention',
  )
  parser.add_argument(
      '--mixed_precision', action='store_true', help='use mixed precision'
  )
  args = parser.parse_args()

  model = torch.nn.DataParallel(RAFT(args))
  model.load_state_dict(torch.load(args.model))
  print(f'Loaded checkpoint at {args.model}')
  flow_model = model.module
  flow_model.cuda()  # .eval()
  flow_model.eval()

  scene_name = args.scene_name
  image_list = sorted(
      glob.glob(os.path.join(args.datapath, '*.png'))
  )  # [::stride]
  image_list += sorted(
      glob.glob(os.path.join(args.datapath, '*.jpg'))
  )  # [::stride]
  
  print(f"Scene: {scene_name}")
  print(f"Data path: {args.datapath}")
  print(f"Found {len(image_list)} images")
  
  if len(image_list) == 0:
    print(f"ERROR: No images found in {args.datapath}")
    print("Please check:")
    print("  1. The datapath is correct")
    print("  2. Images are in PNG or JPG format")
    print("  3. The directory exists and is accessible")
    sys.exit(1)
  
  img_data = []

  for t, (image_file) in tqdm.tqdm(enumerate(image_list)):
    image = cv2.imread(image_file)[..., ::-1]  # rgb
    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
    image = cv2.resize(image, (w1, h1))
    image = image[: h1 - h1 % 8, : w1 - w1 % 8].transpose(2, 0, 1)
    img_data.append(image)

  img_data = np.array(img_data)
  
  print(f"Loaded {img_data.shape[0]} images with shape {img_data.shape}")
  if img_data.shape[0] == 0:
    print("ERROR: No images loaded!")
    sys.exit(1)

  flows_low = []

  flows_high = []
  flow_masks_high = []

  flow_init = None
  flows_arr_low_bwd = {}
  flows_arr_low_fwd = {}

  ii = []
  jj = []
  flows_arr_up = []
  masks_arr_up = []

  for step in [1, 2, 4, 8, 15]:
    start_idx = max(0, -step)
    end_idx = img_data.shape[0] - max(0, step)
    print(f"Step {step}: processing frames from {start_idx} to {end_idx} (total: {max(0, end_idx - start_idx)} pairs)")
    flows_arr_low = []
    for i in tqdm.tqdm(range(max(0, -step), img_data.shape[0] - max(0, step))):
      image1 = (
          torch.as_tensor(np.ascontiguousarray(img_data[i : i + 1]))
          .float()
          .cuda()
      )
      image2 = (
          torch.as_tensor(
              np.ascontiguousarray(img_data[i + step : i + step + 1])
          )
          .float()
          .cuda()
      )

      ii.append(i)
      jj.append(i + step)

      with torch.no_grad():
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        if np.abs(step) > 1:
          flow_init = np.stack(
              [flows_arr_low_fwd[i], flows_arr_low_bwd[i + step]], axis=0
          )
          flow_init = (
              torch.as_tensor(np.ascontiguousarray(flow_init))
              .float()
              .cuda()
              .permute(0, 3, 1, 2)
          )
        else:
          flow_init = None

        flow_low, flow_up, _ = flow_model(
            torch.cat([image1, image2], dim=0),
            torch.cat([image2, image1], dim=0),
            iters=22,
            test_mode=True,
            flow_init=flow_init,
        )

        flow_low_fwd = flow_low[0].cpu().numpy().transpose(1, 2, 0)
        flow_low_bwd = flow_low[1].cpu().numpy().transpose(1, 2, 0)

        # Get target dimensions (half of padded dimensions)
        target_h = flow_up.shape[-2] // 2
        target_w = flow_up.shape[-1] // 2
        
        flow_up_fwd = resize_flow(
            flow_up[0].cpu().numpy().transpose(1, 2, 0),
            target_h,
            target_w,
        )
        flow_up_bwd = resize_flow(
            flow_up[1].cpu().numpy().transpose(1, 2, 0),
            target_h,
            target_w,
        )

        # Verify shapes are consistent
        assert flow_up_fwd.shape == flow_up_bwd.shape, \
            f"Flow shape mismatch: fwd {flow_up_fwd.shape} vs bwd {flow_up_bwd.shape}"
        
        if len(flows_arr_up) > 0 and flows_arr_up[0].shape != flow_up_fwd.shape:
            print(f"Warning: Flow shape changed from {flows_arr_up[0].shape} to {flow_up_fwd.shape}")

        bwd2fwd_flow = warp_flow(flow_up_bwd, flow_up_fwd)
        fwd_lr_error = np.linalg.norm(flow_up_fwd + bwd2fwd_flow, axis=-1)
        fwd_mask_up = fwd_lr_error < 1.0

        # flows_arr_low.append(flow_low_fwd)
        flows_arr_low_bwd[i + step] = flow_low_bwd
        flows_arr_low_fwd[i] = flow_low_fwd

        # masks_arr_low.append(fwd_mask_low)
        flows_arr_up.append(flow_up_fwd)
        masks_arr_up.append(fwd_mask_up)

  iijj = np.stack((ii, jj), axis=0)
  
  # Debug information
  print(f"Total flows collected: {len(flows_arr_up)}")
  if len(flows_arr_up) > 0:
    print(f"First flow shape: {flows_arr_up[0].shape}")
    print(f"Checking all flow shapes are consistent...")
    shapes = [f.shape for f in flows_arr_up]
    unique_shapes = set(shapes)
    if len(unique_shapes) > 1:
      print(f"ERROR: Inconsistent shapes found: {unique_shapes}")
      for idx, shape in enumerate(shapes):
        if shape != shapes[0]:
          print(f"  Flow {idx} has different shape: {shape} (expected {shapes[0]})")
      sys.exit(1)
    else:
      print(f"All flows have consistent shape: {shapes[0]}")
  else:
    print("ERROR: No flows collected!")
    sys.exit(1)
  
  # Convert list to numpy array: shape will be (N, H, W, 2)
  # Use explicit stacking to ensure proper array creation
  try:
    flows_high = np.stack(flows_arr_up, axis=0)
    print(f"flows_high shape after np.stack: {flows_high.shape}")
  except Exception as e:
    print(f"ERROR during np.stack: {e}")
    print(f"Attempting np.array instead...")
    flows_high = np.array(flows_arr_up)
    print(f"flows_high shape after np.array: {flows_high.shape}")
    print(f"flows_high dtype: {flows_high.dtype}")
  
  if flows_high.ndim == 4 and flows_high.shape[-1] == 2:
    # Shape is (N, H, W, 2), transpose to (N, 2, H, W)
    flows_high = flows_high.transpose(0, 3, 1, 2)
    print(f"flows_high shape after transpose: {flows_high.shape}")
  else:
    print(f"ERROR: Expected 4D array with shape (N, H, W, 2), got shape {flows_high.shape}")
    sys.exit(1)
    
  flow_masks_high = np.array(masks_arr_up)[:, None, ...]
  print(f"flow_masks_high shape: {flow_masks_high.shape}")
  
  Path('./cache_flow/%s' % scene_name).mkdir(parents=True, exist_ok=True)
  np.save('./cache_flow/%s/flows.npy' % scene_name, np.float16(flows_high))
  np.save('./cache_flow/%s/flows_masks.npy' % scene_name, flow_masks_high)
  np.save('./cache_flow/%s/ii-jj.npy' % scene_name, iijj)
  
  print(f"Successfully saved flow data for {scene_name}")
  print(f"  - flows.npy: shape {flows_high.shape}, dtype float16")
  print(f"  - flows_masks.npy: shape {flow_masks_high.shape}")
  print(f"  - ii-jj.npy: shape {iijj.shape}")
