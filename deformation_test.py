import contextlib as ctl
import functools as ft
import json
import os
import os.path as osp

import numpy as np
import plotnine as pn
import polars as pl
import tifffile
import torch
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
BG_PATH = "./nersemble-data/018/sequences/BACKGROUND/image_{pose}.jpg"
IMAGE_PATH = "./nersemble-data/018/sequences/EMO-1-shout+laugh/images/cam_{pose}.mp4:{frame}"
DEPTH_PATH = "./depth/NERS-9018_{pose}_depth-{frame}_checkpoint-300000.tiff"
NERS_DFM_PATH = "./deformation/NERS-9018_{pose}_deformation-{frame}_checkpoint-300000.tiff"
UNIM_FLOW_PATH = "./flow/{pose}-{frame}-flow.png"

# output
DFM_PATH = "./deformation-test/{cat}-{pose}-{frame}.png"

CAMERA_PARAMS_PATH = "/home/andrei/adl4cv-project/nersemble-data/018/calibration/camera_params.json"
@ft.cache
def camera_params():
  with open(CAMERA_PARAMS_PATH) as f:
    return json.load(f)

def run(
  *,
  pose_id: str,
  fr0: int,
):
  # pylint: disable=function-redefined,unnecessary-lambda-assignment

  fr1 = fr0 + 1

  fmt = dict(
    pose=pose_id,
  )

  # output_path = DFM_PATH.format(pose=pose_id, frame=fr0)
  # os.makedirs(osp.dirname(output_path), exist_ok=True)
  # if osp.exists(output_path):
  #   return

  # pose
  scale_factor = 9

  # pose = Pose(
  #   camera_params()["world_2_cam"][pose_id],
  #   pose_type=PoseType.WORLD_2_CAM,
  #   camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
  # )
  # pose.change_pose_type(PoseType.CAM_2_WORLD)
  # pose[:3, 3] *= scale_factor

  # intrinsics
  # intr = Intrinsics(camera_params()["intrinsics"])

  # load images
  # bg   = load_img(BG_PATH.format(**fmt), output_type=np.ndarray)
  # rgb0 = load_img(IMAGE_PATH.format(**fmt, frame=fr0), output_type=np.ndarray)
  # rgb1 = load_img(IMAGE_PATH.format(**fmt, frame=fr1), output_type=np.ndarray)
  # dt0  = image_to_depth(DEPTH_PATH.format(**fmt, frame=fr0))[..., None].astype(np.float64)
  # dt1  = image_to_depth(DEPTH_PATH.format(**fmt, frame=fr1))[..., None].astype(np.float64)
  dfm0 = image_to_deformation(NERS_DFM_PATH.format(**fmt, frame=fr0)).astype(np.float64)
  dfm1 = image_to_deformation(NERS_DFM_PATH.format(**fmt, frame=fr1)).astype(np.float64)
  # unimfl0 = image_to_flow_hue(UNIM_FLOW_PATH.format(**fmt, frame=fr0))

  delta = dfm1 - dfm0

  array_to_normed_image(DFM_PATH.format(cat="dfm", pose=pose_id, frame=fr0), dfm0)
  array_to_normed_image(DFM_PATH.format(cat="delta", pose=pose_id, frame=fr0), delta)

  # a_mag = pipe(
  #   delta,
  #   lambda a: np.linalg.norm(a, ord=2, axis=-1, keepdims=True),
  #   # lambda a: np.max(a, axis=tuple(i for i in range(len(a.shape)))[:-1], keepdims=True),
  # )
  # print(a_mag)
  # if True:
  #   df = pl.DataFrame(dict(mag=np.ravel(a_mag)))
  #   (pn.ggplot(df, pn.aes(x="mag"))
  #     + pn.geom_density()
  #   ).save(DFM_PATH.format(cat="plot", pose=pose_id, frame=fr0))

  # Image.fromarray(((normify(delta) + 1) * 255).astype(np.uint8)).save(DFM_PATH.format(cat="dfm-delta", pose=pose_id, frame=fr0))
  # Image.fromarray(((normify(dfm0) + 1) * 255).astype(np.uint8)).save(DFM_PATH.format(cat="dfm", pose=pose_id, frame=fr0))

  # delta = dt1 - dt0
  # Image.fromarray(((normify(delta) + 1) * 255).astype(np.uint8)).save(output_path)

  # tifffile.imwrite(
  #   output_path,
  #   delta.astype(np.float16).view(np.int16),
  #   compression="ZLIB",
  #   photometric="rgb",
  # )

def main():
  pose_ids = [
    # "221501007",

    "220700191",
    # "222200036",
    # "222200037",
    # "222200038",
    # "222200039",
    # "222200040",
    # "222200041",
    # "222200042",
    # "222200043",
    # "222200044",
    # "222200045",
    # "222200046",
    # "222200047",
    # "222200048",
    # "222200049",
  ]

  pose_id = "220700191"

  ema = None

  for fr0 in tqdm(range(1, 412+1)):
    fr1 = fr0 + 1

    dfm0 = image_to_deformation(NERS_DFM_PATH.format(pose=pose_id, frame=fr0)).astype(np.float64)
    dfm1 = image_to_deformation(NERS_DFM_PATH.format(pose=pose_id, frame=fr1)).astype(np.float64)

    is_bg = (
        (np.linalg.norm(dfm0, axis=-1, ord=2) <= 0.01)
      | (np.linalg.norm(dfm1, axis=-1, ord=2) <= 0.01)
    )

    delta = dfm1 - dfm0
    # if ema is None:
    #   ema = delta
    # else:
    #   alpha = 0.6
    #   ema = ema * (1 - alpha) + delta * alpha
    # array_to_normed_image(DFM_PATH.format(cat="ema", pose=pose_id, frame=fr0), ema)

    
    # array_to_normed_image(DFM_PATH.format(cat="dfm", pose=pose_id, frame=fr0), dfm0)
    # array_to_normed_image(DFM_PATH.format(cat="delta", pose=pose_id, frame=fr0), delta)

  # for fr0 in [154]:
  # for fr0 in range(140, 160):
    # for pose_id in pose_ids:
    #   # print(fr0)
    #   run(
    #     pose_id=pose_id,
    #     fr0=fr0,
    #   )

if __name__ == "__main__":
  main()

