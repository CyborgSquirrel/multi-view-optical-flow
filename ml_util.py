import dataclasses as dc
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

try:
  import unimatch_utils.frame_utils
except ModuleNotFoundError:
  logger.warning("Unimatch not setup, can't import it...")

def unimatch_load_img(img: str | Path) -> torch.Tensor:
  if isinstance(img, Path):
    img = str(img)
  if isinstance(img, str):
    img = unimatch_utils.frame_utils.read_gen(img)
  if isinstance(img, Image.Image):
    img = img.convert('RGB')
    img = np.array(img).astype(np.uint8)
  if isinstance(img, np.ndarray):
    dtype = np.uint8
    if img.dtype != dtype:
      raise TypeError(f'Expected {dtype=}, got {img.dtype=}')
  img = img[..., :3]  # [H, W, C]
  img = torch.from_numpy(img).permute(2, 0, 1).float()  # [C, H, W]
  return img

def unimatch_get_transform(
  imgs: list[torch.Tensor],  # [C, H, W]
  *,
  padding_factor=8,
  infer_size=None,
  infer_scale=None,
) -> tuple[
  Callable[[torch.Tensor], torch.Tensor],
  Callable[[torch.Tensor], torch.Tensor],
]:
  orig_shape = tuple(imgs[0].shape)
  orig_size = orig_shape[-2:]
  if not all(img.shape == orig_shape for img in imgs):
    raise RuntimeError("Shapes of all images must be equal")

  # Resize to nearest size or specified size
  if infer_size is None:
    size = orig_size

    # Scale size
    if infer_scale is not None:
      size = (
        size[-2] * infer_scale,
        size[-1] * infer_scale,
      )

    # Snap size to padding factor
    size = (
      int(np.ceil(size[-2] / padding_factor)) * padding_factor,
      int(np.ceil(size[-1] / padding_factor)) * padding_factor,
    )

    infer_size = size
  assert isinstance(infer_size, tuple)

  # The model is trained with size: width > height
  inference_transpose = False
  if orig_shape[-2] > orig_shape[-1]:
    inference_transpose = True

  def trans_fn(img: torch.Tensor):
    # Resize
    if infer_size != orig_size:
      img = img.unsqueeze(0)
      img = F.interpolate(
        img,
        size=infer_size,
        mode='bilinear',
        align_corners=True,
      )
      img = img.squeeze(0)

    # Transpose
    if inference_transpose:
      img = torch.transpose(img, -2, -1)

    return img

  def restore_fn(img: torch.Tensor, *, scale_flow: bool):
    # Transpose
    if inference_transpose:
      img = torch.transpose(img, -2, -1)

    # Resize
    if infer_size != orig_size:
      img = img.unsqueeze(0)
      img = F.interpolate(
        img,
        size=orig_size,
        mode='bilinear',
        align_corners=True,
      )
      img = img.squeeze(0)

      # NOTE(andrei): I _think_ this adapts the flow to the new size, but
      # honestly not 100% sure...
      if scale_flow:
        img[:, 0] = img[:, 0] * orig_size[-1] / infer_size[-1]
        img[:, 1] = img[:, 1] * orig_size[-2] / infer_size[-2]

    return img

  return trans_fn, restore_fn

def flow_to_image_rgb(
  # [H, W, 2]
  flow: torch.Tensor | np.ndarray,
) -> Image.Image:
  shape = tuple(flow.shape)
  match shape:
    case (_h, _w, 2): pass
    case _: raise TypeError(f"Unexpected {shape=}")

  if isinstance(flow, torch.Tensor):
    flow = flow.cpu().numpy()

  thresh = np.max(np.abs(flow), axis=(0, 1))

  lo = -thresh
  hi =  thresh

  flow = (flow - lo) / (hi - lo)
  flow = flow * 128 + 128
  flow = np.stack([
    flow[..., 0],
    np.zeros(shape[:-1]),
    flow[..., 1],
  ], axis=-1)

  flow = np.clip(flow, 0, 255)
  flow = flow.astype(np.uint8)

  flow = Image.fromarray(flow)
  return flow

def hsv_image_to_rgb(
  # [H, W, 3]
  hsv: np.ndarray,
) -> np.ndarray:
  h = hsv[..., 0]
  s = hsv[..., 1]
  v = hsv[..., 2]

  h6 = h * 6.0
  i = np.floor(h6).astype(np.int8)
  f = h6 - i

  p = v * (1 - s)
  q = v * (1 - f * s)
  t = v * (1 - (1 - f) * s)

  i_mod = (i % 6)[..., None]  # [..., 1]

  # Prebuild the 6 possible RGB triplets
  candidates = np.stack([
    np.stack([v, t, p], axis=-1),
    np.stack([q, v, p], axis=-1),
    np.stack([p, v, t], axis=-1),
    np.stack([p, q, v], axis=-1),
    np.stack([t, p, v], axis=-1),
    np.stack([v, p, q], axis=-1),
  ], axis=-2)   # [..., 6, 3]

  # Pick the correct one for each pixel
  rgb = np.take_along_axis(candidates, i_mod[..., None], axis=-2)[..., 0, :]

  return rgb

def flow_to_image_hue(
  # [H, W, 2]
  flow: torch.Tensor | np.ndarray,
) -> Image.Image:
  shape = tuple(flow.shape)
  match shape:
    case (_h, _w, 2): pass
    case _: raise TypeError(f"Unexpected {shape=}")

  if isinstance(flow, torch.Tensor):
    flow = flow.cpu().numpy()

  ang = np.atan2(flow[..., 0], flow[..., 1])
  ang = (np.pi + ang) / (np.pi * 2)

  mag = np.linalg.vector_norm(flow, axis=-1, ord=2)
  mag = mag / np.max(mag)

  flow = np.stack([
    ang,
    mag,
    np.ones(shape[:-1]),
  ], axis=-1)
  flow = hsv_image_to_rgb(flow)
  flow = np.clip(flow * 255, 0, 255).astype(np.uint8)

  flow = Image.fromarray(flow)
  return flow

def forward_warp(im0, flow, interpolation_mode):
  im0 = im0.to(torch.float32)
  im1 = torch.zeros_like(im0)
  B = im0.shape[0]
  H = im0.shape[2]
  W = im0.shape[3]
  if interpolation_mode == 0:
    for b in range(B):
      for h in range(H):
        for w in range(W):
          x = w + flow[b, h, w, 0]
          y = h + flow[b, h, w, 1]
          nw = (int(torch.floor(x)), int(torch.floor(y)))
          ne = (nw[0]+1, nw[1])
          sw = (nw[0], nw[1]+1)
          se = (nw[0]+1, nw[1]+1)
          p = im0[b, :, h, w]

          if nw[0] >= 0 and se[0] < W and nw[1] >= 0 and se[1] < H:
            nw_k = (se[0]-x)*(se[1]-y)
            ne_k = (x-sw[0])*(sw[1]-y)
            sw_k = (ne[0]-x)*(y-ne[1])
            se_k = (x-nw[0])*(y-nw[1])
            im1[b, :, nw[1], nw[0]] += nw_k*p
            im1[b, :, ne[1], ne[0]] += ne_k*p
            im1[b, :, sw[1], sw[0]] += sw_k*p
            im1[b, :, se[1], se[0]] += se_k*p
  else:
    round_flow = torch.round(flow)
    for b in range(B):
      for h in range(H):
        for w in range(W):
          x = w + int(round_flow[b, h, w, 0])
          y = h + int(round_flow[b, h, w, 1])
          if x >= 0 and x < W and y >= 0 and y < H:
            im1[b, :, y, x] = im0[b, :, h, w]
  im1 = torch.clip(im1, 0, 255)
  return im1
