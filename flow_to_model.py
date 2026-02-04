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
from PIL import Image
from scipy.interpolate import griddata
from skimage.transform import resize  # pylint: disable
from tqdm import tqdm

from ml_util import (depth_to_image, flow_to_hue_flow, flow_to_image_hue,
                     image_to_depth, image_to_flow_hue, load_img, normify,
                     numpify)
from unimatch.unimatch import flow_warp

EXP_NAME = "warped-flow-2d-linear"

IMAGE_PATH = "./nersemble-data/018/sequences/EMO-1-shout+laugh/images/cam_{pose}.mp4:{frame}"
DEPTH_PATH = "./depth/NERS-9018_{pose}_depth-{frame}_checkpoint-300000.tiff"
BG_PATH = "./nersemble-data/018/sequences/BACKGROUND/image_{pose}.jpg"
FLOW_PATH = "./flow/{pose}-{frame}-flow.png"

MODEL_PATH = "./model/{output}-{pose}-{frame}.ply"

WARPED_2D_FLOW_PATH = "./{exp}/{pose}-{frame}.png"

CAMERA_PARAMS_PATH = "/home/andrei/adl4cv-project/nersemble-data/018/calibration/camera_params.json"
@ft.cache
def camera_params():
  with open(CAMERA_PARAMS_PATH) as f:
    return json.load(f)

def make_flow(
  *,
  pose_id: str,
  fr0: int,
):
  fr1 = fr0 + 1

  fmt = dict(
    pose=pose_id,
  )

  # pose
  scale_factor = 9

  pose = Pose(
    camera_params()["world_2_cam"][pose_id],
    pose_type=PoseType.WORLD_2_CAM,
    camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
  )
  pose.change_pose_type(PoseType.CAM_2_WORLD)
  pose[:3, 3] *= scale_factor

  # intrinsics
  intr = Intrinsics(camera_params()["intrinsics"])

  # load images
  bg = load_img(BG_PATH.format(**fmt), output_type=np.ndarray)
  flow = image_to_flow_hue(FLOW_PATH.format(**fmt, frame=fr0))

  im0 = load_img(IMAGE_PATH.format(**fmt, frame=fr0), output_type=np.ndarray)
  im1 = load_img(IMAGE_PATH.format(**fmt, frame=fr1), output_type=np.ndarray)

  dt0 = image_to_depth(DEPTH_PATH.format(**fmt, frame=fr0))[..., None].astype(np.float64)
  dt1 = image_to_depth(DEPTH_PATH.format(**fmt, frame=fr1))[..., None].astype(np.float64)

  # resize images
  shape = dt0.shape[:2]

  im0 = resize(im0, shape, preserve_range=True)
  im1 = resize(im1, shape, preserve_range=True)
  bg = resize(bg, shape, preserve_range=True)

  # rgb
  rgb0 = im0 / 255
  rgb1 = im1 / 255
  bg = bg / 255

  # image plane
  h, w = shape
  mul = 4
  x, y = np.meshgrid(np.arange(w) * mul, np.arange(h) * mul)
  xy = np.stack([x, y], axis=-1)

  # flow
  xy0 = xy + flow * mul
  # xy0 = xy + flow
  xy1 = xy

  # warp depth
  # flow_warp_np = numpify(flow_warp)
  # dt1 = np.permute_dims(flow_warp_np(
  #   np.permute_dims(dt0, (2, 0, 1))[None, ...],
  #   np.permute_dims(flow, (2, 0, 1))[None, ...],
  # )[0, ...], (1, 2, 0))

  # flatten
  xy   = xy  .reshape(-1, xy  .shape[-1])
  xy0  = xy0 .reshape(-1, xy0 .shape[-1])
  xy1  = xy1 .reshape(-1, xy1 .shape[-1])
  dt1  = dt1 .reshape(-1, dt1 .shape[-1])
  dt0  = dt0 .reshape(-1, dt0 .shape[-1])
  rgb0 = rgb0.reshape(-1, rgb0.shape[-1])
  rgb1 = rgb1.reshape(-1, rgb1.shape[-1])
  bg   = bg  .reshape(-1, bg  .shape[-1])

  # remove background
  if True:
  # if False:
    is_bg = (
        (np.linalg.norm(rgb0 - bg, axis=-1, ord=2) <= 0.2)
      | (np.linalg.norm(rgb1 - bg, axis=-1, ord=2) <= 0.2)
    )
    xy0  = xy0 [~is_bg]
    xy1  = xy1 [~is_bg]
    dt0  = dt0 [~is_bg]
    dt1  = dt1 [~is_bg]
    rgb0 = rgb0[~is_bg]
    rgb1 = rgb1[~is_bg]

  # 2D homogeneous
  xyz0 = np.concat([xy0, np.ones( xy0.shape[:-1] + (1,) )], axis=-1)
  xyz1 = np.concat([xy1, np.ones( xy1.shape[:-1] + (1,) )], axis=-1)

  # depth
  xyz0 = xyz0 * dt0
  xyz1 = xyz1 * dt0
  # xyz1 = xyz1 * dt1

  # intrinsics
  xyz0 = np.einsum("ij,aj->ai", np.linalg.inv(intr), xyz0)
  xyz1 = np.einsum("ij,aj->ai", np.linalg.inv(intr), xyz1)

  # if True:
  if False:
    # 3D homogeneous
    xyz0 = np.concat([xyz0, np.ones( xyz0.shape[:-1] + (1,) )], axis=-1)
    xyz1 = np.concat([xyz1, np.ones( xyz1.shape[:-1] + (1,) )], axis=-1)

    # pose
    xyz0 = np.einsum("ij,aj->ai", pose, xyz0)
    xyz1 = np.einsum("ij,aj->ai", pose, xyz1)

    # 3D dehomogenize
    xyz0 = xyz0[..., :3] / xyz0[[3]]
    xyz1 = xyz1[..., :3] / xyz1[[3]]

  flow3 = -(xyz0 - xyz1)

  if True:
    df = pl.concat([
      pl.DataFrame(dict(v=flow3[..., 0], c="x")),
      pl.DataFrame(dict(v=flow3[..., 1], c="y")),
      pl.DataFrame(dict(v=flow3[..., 2], c="z")),
    ])
    (pn.ggplot(df, pn.aes(x="v", color="c"))
      + pn.geom_density()
    ).show()

  # get scene flow
  # if True:
  #   flow3 = xyz0 - xyz1

  #   # flow3_amax = np.max(np.abs(flow3))
  #   # flow3 = ((flow3 / flow3_amax) + 1) / 2



  # _, flow_hue = flow_to_hue_flow(flow2)
  # print(flow_hue.shape)

  # trimesh.points.PointCloud(
  #   np.concat([ xy1, np.zeros( xy1.shape[:-1] )[..., None] ], axis=-1),
  #   colors=flow_hue,
  # ).export(MODEL_PATH.format(output="flow", pose=pose_id))

  # trimesh.points.PointCloud(
  #   np.concat([ xy1, np.zeros( xy1.shape[:-1] )[..., None] ], axis=-1),
  #   colors=rgb1,
  # ).export(MODEL_PATH.format(output="color", pose=pose_id))



  trimesh.points.PointCloud(
    xyz1,
    colors=normify(flow3),
  ).export(MODEL_PATH.format(output="flow3", pose=pose_id, frame=fr0))

def main():
  pose_ids = [
    "221501007",

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

  # for fr0 in tqdm(range(1, 413+1)):
  for fr0 in [250]:
    for pose_id in pose_ids:
      make_flow(
        pose_id=pose_id,
        fr0=fr0,
      )

if __name__ == "__main__":
  main()
