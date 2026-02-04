import contextlib as ctl
import functools as ft
import json
import os
import os.path as osp
import shutil
from pathlib import Path

import numpy as np
import plotnine as pn
import polars as pl
import torch
import torch.nn.functional as F
import trimesh
from dreifus.matrix import (CameraCoordinateConvention, Intrinsics, Pose,
                            PoseType)
from PIL import Image
from scipy.interpolate import griddata
from skimage.transform import resize  # pylint: disable
from tqdm import tqdm

from ml_util import (array_to_normed_image, deformation_to_image,
                     depth_to_image, flow_to_hue_flow, flow_to_image_hue,
                     image_to_deformation, image_to_depth, image_to_flow_hue,
                     load_img, normify, numpify)
from unimatch.unimatch import flow_warp
from util import pipe

# input
IMAGE_PATH = "./nersemble-data/018/sequences/EMO-1-shout+laugh/images/cam_{pose}.mp4:{frame}"
FLOW_NERS_PATH = "./ners-flow/{pose}-{frame}.png"
FLOW_UNIM_PATH = "./flow/{pose}-{frame}-flow.png"

# output
OUT_PATH_PNG = "./output/{output}-{pose}-{frame}.png"
OUT_PATH_TXT = "./output/{output}-{pose}-{frame}.txt"

def main():
  os.makedirs(osp.dirname(OUT_PATH_PNG), exist_ok=True)

  pose_ids = [
    "222200044",
    "222200038",
    "222200045",
    "222200041",
    # "222200046",
    "222200039",
    "221501007",
    "222200037",
    "222200040",
    # "220700191",
    "222200036",
    "222200043",
  ]

  # pose_id = "220700191"

  # for fr0 in tqdm(range(1, 412)):
  for fr0 in tqdm(range(106, 412)):
    for pose_id in pose_ids:
    # for fr0 in [270]:
    # for fr0 in range(265, 275):
      flow_ners = image_to_flow_hue(FLOW_NERS_PATH.format(pose=pose_id, frame=fr0))
      flow_ners = flow_ners / 8
      flow_ners = torch.from_numpy(flow_ners).permute(2, 0, 1).unsqueeze(0)

      flow_unim = image_to_flow_hue(FLOW_UNIM_PATH.format(pose=pose_id, frame=fr0))
      flow_unim = flow_unim / 4
      flow_unim = torch.from_numpy(flow_unim).permute(2, 0, 1).unsqueeze(0)

      shape = flow_ners.shape[-2:]

      img0 = load_img(IMAGE_PATH.format(pose=pose_id, frame=fr0-1))
      img0 = img0.permute(2, 0, 1).unsqueeze(0)
      img0 = F.interpolate(
        img0,
        size=shape,
        mode="bilinear",
        align_corners=True,
      )

      img_warp_ners = flow_warp(
        img0.float(),
        flow_ners.float(),
      )

      img_warp_unim = flow_warp(
        img0.float(),
        flow_unim.float(),
      )

      img0 = pipe(
        img0,
        lambda a: a.squeeze(0).permute(1, 2, 0),
        lambda a: torch.round(a),
        lambda a: torch.clip(a, 0, 255),
        lambda a: a.to(torch.uint8),
        lambda a: a.numpy(),
      )

      img_warp_ners = pipe(
        img_warp_ners,
        lambda a: a.squeeze(0).permute(1, 2, 0),
        lambda a: torch.round(a),
        lambda a: torch.clip(a, 0, 255),
        lambda a: a.to(torch.uint8),
        lambda a: a.numpy(),
      )
      Image.fromarray(img_warp_ners).save(OUT_PATH_PNG.format(output="img-warp-ners", pose=pose_id, frame=fr0))

      img_warp_unim = pipe(
        img_warp_unim,
        lambda a: a.squeeze(0).permute(1, 2, 0),
        lambda a: torch.round(a),
        lambda a: torch.clip(a, 0, 255),
        lambda a: a.to(torch.uint8),
        lambda a: a.numpy(),
      )
      Image.fromarray(img_warp_unim).save(OUT_PATH_PNG.format(output="img-warp-unim", pose=pose_id, frame=fr0))

      img1 = pipe(
        load_img(IMAGE_PATH.format(pose=pose_id, frame=fr0)),
        lambda a: a.permute(2, 0, 1).unsqueeze(0),
        lambda a: F.interpolate(
          a,
          size=shape,
          mode="bilinear",
          align_corners=True,
        ),
        lambda a: a.squeeze(0).permute(1, 2, 0),
        lambda a: torch.round(a),
        lambda a: torch.clip(a, 0, 255),
        lambda a: a.to(torch.uint8),
        lambda a: a.numpy(),
      )
      Image.fromarray(img1).save(OUT_PATH_PNG.format(output="img-gt", pose=pose_id, frame=fr0))

      # err base
      err_base = np.mean(np.abs(img1.astype(np.float64) - img0.astype(np.float64)), axis=-1)
      Path(OUT_PATH_TXT.format(output="err-base", pose=pose_id, frame=fr0)).write_text(f"{np.mean(err_base)}\n")

      err_base = pipe(
        err_base,
        lambda a: np.round(a),
        lambda a: np.clip(a, 0, 255),
        lambda a: a.astype(np.uint8),
      )
      Image.fromarray(err_base).save(OUT_PATH_PNG.format(output="err-base", pose=pose_id, frame=fr0))

      # err ners
      err_ners = np.mean(np.abs(img1.astype(np.float64) - img_warp_ners.astype(np.float64)), axis=-1)
      Path(OUT_PATH_TXT.format(output="err-ners", pose=pose_id, frame=fr0)).write_text(f"{np.mean(err_ners)}\n")

      err_ners = pipe(
        err_ners,
        lambda a: np.round(a),
        lambda a: np.clip(a, 0, 255),
        lambda a: a.astype(np.uint8),
      )
      Image.fromarray(err_ners).save(OUT_PATH_PNG.format(output="err-ners", pose=pose_id, frame=fr0))

      # err unim
      err_unim = np.mean(np.abs(img1.astype(np.float64) - img_warp_unim.astype(np.float64)), axis=-1)
      Path(OUT_PATH_TXT.format(output="err-unim", pose=pose_id, frame=fr0)).write_text(f"{np.mean(err_unim)}\n")

      err_unim = pipe(
        err_unim,
        lambda a: np.round(a),
        lambda a: np.clip(a, 0, 255),
        lambda a: a.astype(np.uint8),
      )
      Image.fromarray(err_unim).save(OUT_PATH_PNG.format(output="err-unim", pose=pose_id, frame=fr0))

if __name__ == "__main__":
  main()
