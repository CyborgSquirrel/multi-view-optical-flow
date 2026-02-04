import contextlib as ctl
import functools as ft
import json
import os
import os.path as osp

import numpy as np
import plotnine as pn
import polars as pl
import torch
import trimesh
from dreifus.matrix import (CameraCoordinateConvention, Intrinsics, Pose,
                            PoseType)
from scipy.interpolate import griddata
from skimage.transform import resize  # pylint: disable
from tqdm import tqdm

from ml_util import (array_to_mag_image, array_to_normed_image, depth_to_image,
                     flow_to_hue_flow, flow_to_image_hue, image_to_deformation,
                     image_to_depth, image_to_flow_hue, load_img, normify,
                     numpify)
from unimatch.unimatch import flow_warp
from util import pipe

LIN_FLOW_PATH = "./warped-flow-2d-linear/{pose}-{frame}.png"
BIN_FLOW_PATH = "./warped-flow-2d-binned/{pose}-{frame}.png"
UNIM_FLOW_PATH = "./flow/{pose}-{frame}-flow.png"

ERR_PATH = "./flow-err/{output}-{pose}-{frame}.png"

def main():
  pose_ids = [
    # "221501007",

    "220700191",
    "222200036",
    "222200037",
    "222200038",
    "222200039",
    "222200040",
    "222200041",
    "222200042",
    "222200043",
    "222200044",
    "222200045",
    "222200046",
    "222200047",
    "222200048",
    "222200049",
  ]

  dst_pose_id = "221501007"

  # def get_mask(a):
  #   return np.linalg.norm(a, ord=2, axis=-1) > 0.1

  # def get_err(a, b, *, mask):
  #   err = np.zeros(a.shape[:-1] + (1,))
  #   err[mask] = np.mean(np.abs(a - b), axis=-1, keepdims=True)[mask]
  #   return err

  def get_err(a, b):
    return np.mean(np.abs(a - b), axis=-1, keepdims=True)

  for fr0 in tqdm(range(1, 413)):
  # for fr0 in range(1, 413):
  # for fr0 in [270]:
    unim_flow = image_to_flow_hue(UNIM_FLOW_PATH.format(pose=dst_pose_id, frame=fr0))
    # print(np.mean(np.linalg.norm(unim_flow, axis=-1)))
    # unim_mask = get_mask(unim_flow)

    # array_to_mag_image(ERR_PATH.format(output="mask", pose=dst_pose_id, frame=fr0), unim_mask[..., None])

    for pose_id in pose_ids:
      scale = 4

      lin_flow = image_to_flow_hue(LIN_FLOW_PATH.format(pose=pose_id, frame=fr0)) / scale
      # print(np.mean(np.linalg.norm(lin_flow, axis=-1)))
      lin_err = get_err(
        unim_flow,
        lin_flow,
        # mask=unim_mask | get_mask(lin_flow),
      )
      array_to_mag_image(ERR_PATH.format(output="lin", pose=pose_id, frame=fr0), lin_err)
      # print("lin_err", np.mean(lin_err))

      bin_flow = image_to_flow_hue(BIN_FLOW_PATH.format(pose=pose_id, frame=fr0)) / scale
      bin_err = get_err(
        unim_flow,
        bin_flow,
        # mask=unim_mask | get_mask(bin_flow),
      )
      array_to_mag_image(ERR_PATH.format(output="bin", pose=pose_id, frame=fr0), bin_err)
      # print("bin_err", np.mean(bin_err))

if __name__ == "__main__":
  main()
