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

from ml_util import (depth_to_image, flow_to_hue_flow, flow_to_image_hue,
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
MODEL_PATH = "./model/{output}_{pose}_{frame}.ply"
FLOW_PATH = "./ners-flow/{pose}-{frame}.png"

CAMERA_PARAMS_PATH = "/home/andrei/adl4cv-project/nersemble-data/018/calibration/camera_params.json"
@ft.cache
def camera_params():
  with open(CAMERA_PARAMS_PATH) as f:
    return json.load(f)

DELTA_EMA = dict()

def make_flow(
  *,
  pose_id: str,
  fr0: int,
):
  # pylint: disable=function-redefined,unnecessary-lambda-assignment

  fr1 = fr0 + 1

  fmt = dict(
    pose=pose_id,
  )

  output_path = FLOW_PATH.format(pose=pose_id, frame=fr0)
  os.makedirs(osp.dirname(output_path), exist_ok=True)
  # if osp.exists(output_path):
  #   return

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
  bg   = load_img(BG_PATH.format(**fmt), output_type=np.ndarray)
  rgb0 = load_img(IMAGE_PATH.format(**fmt, frame=fr0), output_type=np.ndarray)
  rgb1 = load_img(IMAGE_PATH.format(**fmt, frame=fr1), output_type=np.ndarray)
  dt0  = image_to_depth(DEPTH_PATH.format(**fmt, frame=fr0))[..., None].astype(np.float64)
  dfm0 = image_to_deformation(NERS_DFM_PATH.format(**fmt, frame=fr0)).astype(np.float64)
  dfm1 = image_to_deformation(NERS_DFM_PATH.format(**fmt, frame=fr1)).astype(np.float64)
  # dfm1 = image_to_deformation(NERS_DFM_PATH.format(**fmt, frame=fr1-2)).astype(np.float64)
  unimfl0 = image_to_flow_hue(UNIM_FLOW_PATH.format(**fmt, frame=fr0))

  # resize images
  shape = dt0.shape[:2]

  # deal with resize + rescale rgb
  fn = lambda a: resize(a, shape, preserve_range=True) / 255
  rgb0 = pipe(rgb0, fn)
  rgb1 = pipe(rgb1, fn)
  bg   = pipe(bg  , fn)

  # image plane
  h, w = shape
  mul = 4
  x, y = np.meshgrid(np.arange(w) * mul, np.arange(h) * mul)
  xy = np.stack([x, y], axis=-1)

  # flow
  xy0 = xy

  # warp depth
  # flow_warp_np = numpify(flow_warp)
  # dt1 = np.permute_dims(flow_warp_np(
  #   np.permute_dims(dt0, (2, 0, 1))[None, ...],
  #   np.permute_dims(flow, (2, 0, 1))[None, ...],
  # )[0, ...], (1, 2, 0))

  # flatten
  fn = lambda a: a.reshape(-1, a.shape[-1])
  xy      = pipe(xy     , fn)
  bg      = pipe(bg     , fn)
  xy0     = pipe(xy0    , fn)
  dt0     = pipe(dt0    , fn)
  rgb0    = pipe(rgb0   , fn)
  rgb1    = pipe(rgb1   , fn)
  dfm0    = pipe(dfm0   , fn)
  dfm1    = pipe(dfm1   , fn)
  unimfl0 = pipe(unimfl0, fn)

  # if True:
  #   df = pl.DataFrame(dict(mag=np.linalg.norm(dfm0, axis=-1)))
  #   (pn.ggplot(df, pn.aes(x="mag"))
  #     + pn.geom_density()
  #   ).show()

  xy00 = xy0

  with ctl.ExitStack() as stack:
    # depth
    xyz0 = np.concat([xy0, np.ones( xy0.shape[:-1] + (1,) )], axis=-1)
    xyz0 = xyz0 * dt0

    @stack.callback
    def cb():
      nonlocal xy0
      xy0 = xyz0[..., :2] / xyz0[..., [2]]

    # intrinsics
    xyz0 = np.einsum("ij,aj->ai", np.linalg.inv(intr), xyz0)

    @stack.callback
    def cb():
      nonlocal xyz0
      xyz0 = np.einsum("ij,aj->ai", intr, xyz0)

    # pose
    # xyzw0 = np.concat([ xyz0, np.ones( xyz0.shape[:-1] + (1,) ) ], axis=-1)
    # xyzw0 = np.einsum("ij,aj->ai", pose, xyzw0)
    # xyz0 = xyzw0[..., :3]

    # @stack.callback
    # def cb():
    #   nonlocal xyz0
    #   xyzw0 = np.concat([ xyz0, np.ones( xyz0.shape[:-1] + (1,) ) ], axis=-1)
    #   xyzw0 = np.einsum("ij,aj->ai", np.linalg.inv(pose), xyzw0)
    #   xyz0 = xyzw0[..., :3]

    # deformation
    # a_xyz0 = xyz0
    # xyz0 = xyz0 - dfm0
    # b_xyz0 = xyz0

    delta = dfm1 - dfm0
    if pose_id not in DELTA_EMA:
      DELTA_EMA[pose_id] = dict(ema=delta)
    else:
      assert DELTA_EMA[pose_id]["frame"] == fr0-1

      alpha = 0.6
      DELTA_EMA[pose_id]["ema"] = (1 - alpha) * DELTA_EMA[pose_id]["ema"] + alpha * delta
    DELTA_EMA[pose_id]["frame"] = fr0

    delta_ema = DELTA_EMA[pose_id]["ema"]

    # delta = delta * np.array([4, 4, 1])[None, ...]

    a_xyz0 = xyz0
    xyz0 = xyz0 + delta_ema
    b_xyz0 = xyz0

    # sort by increasing z axis
    # sort_idx = np.argsort(xyz0[..., 2])
    # xyz0 = xyz0[sort_idx]

  # get scene flow
  # if True:
  #   flow3 = xyz0 - xyz1

  #   # flow3_amax = np.max(np.abs(flow3))
  #   # flow3 = ((flow3 / flow3_amax) + 1) / 2

  # remove background
  if True:
  # if False:
    # is_bg = (
    #     (np.linalg.norm(rgb0 - bg, axis=-1, ord=2) <= 0.2)
    #   | (np.linalg.norm(rgb1 - bg, axis=-1, ord=2) <= 0.2)
    # )
    is_bg = (
        (np.linalg.norm(dfm0, axis=-1, ord=2) <= 0.01)
      | (np.linalg.norm(dfm1, axis=-1, ord=2) <= 0.01)
    )

    fn = lambda a: a[~is_bg]
    xy0     = pipe(xy0    , fn)
    xy00    = pipe(xy00   , fn)
    rgb0    = pipe(rgb0   , fn)
    rgb1    = pipe(rgb1   , fn)
    # dfm0    = pipe(dfm0   , fn)
    # dfm1    = pipe(dfm1   , fn)
    # unimfl0 = pipe(unimfl0, fn)

  flow2 = -(xy00 - xy0)
  # print(np.max(np.linalg.norm(flow2, axis=-1)))
  # flow2 = delta[..., :2]

  # if True:
  if False:
    flow2_mag = np.linalg.norm(flow2, axis=-1, keepdims=True)
    flow2 = flow2 / flow2_mag

    flow2_mag_limit = np.percentile(flow2_mag, 90)
    # flow2_mag_limit = 1.5
    flow2_mag = np.clip(flow2_mag, None, flow2_mag_limit)
    flow2 = flow2 * flow2_mag

    # df = pl.concat([
    #   pl.DataFrame(dict(src="nersemble", mag=np.linalg.norm(flow2, axis=-1))),
    #   pl.DataFrame(dict(src="unimatch", mag=np.linalg.norm(unimfl0, axis=-1))),
    # ])

    # (pn.ggplot(df, pn.aes(x="mag", color="src"))
    #   + pn.geom_density()
    # ).show()

  # if False:
  if True:
    ### warp with binning
    flow_img = np.zeros(shape + (2,))
    xym = (xy00 // mul).astype(int)
    # mask = (
    #     (xym[...,0] >= 0) & (xym[...,0] < shape[1])
    #   & (xym[...,1] >= 0) & (xym[...,1] < shape[0])
    # )
    # flow2 = flow2[mask]
    # xym = xym[mask]
    flow_img[xym[...,1], xym[...,0]] = flow2
    flow_img = flow_to_image_hue(flow_img)
    flow_img.save(output_path, exif=flow_img.getexif())

  if False:
    # _, flow_hue = flow_to_hue_flow(flow2)
    # print(flow_hue.shape)

    # trimesh.points.PointCloud(
    #   np.concat([ xy1, np.zeros( xy1.shape[:-1] )[..., None] ], axis=-1),
    #   colors=flow_hue,
    # ).export(MODEL_PATH.format(output="flow", pose=pose_id))

    trimesh.points.PointCloud(
      a_xyz0,
      colors=rgb0,
    ).export(MODEL_PATH.format(output="color0", pose=pose_id, frame=fr0))

    trimesh.points.PointCloud(
      b_xyz0,
      colors=rgb0,
    ).export(MODEL_PATH.format(output="color1", pose=pose_id, frame=fr0))

    # print(dfm0.shape)

    trimesh.points.PointCloud(
      a_xyz0,
      colors=normify(dfm0),
    ).export(MODEL_PATH.format(output="dfm", pose=pose_id, frame=fr0))

    trimesh.points.PointCloud(
      a_xyz0,
      colors=normify(dfm1),
    ).export(MODEL_PATH.format(output="dfm", pose=pose_id, frame=fr1))

    # HSV 2D ners flow
    flow2_mag_max, flow2_hue = flow_to_hue_flow(flow2)
    # print(flow2_mag_max)

    trimesh.points.PointCloud(
      a_xyz0,
      colors=flow2_hue,
    ).export(MODEL_PATH.format(output="ners_flow_hsv", pose=pose_id, frame=fr0))

    trimesh.points.PointCloud(
      a_xyz0,
      colors=np.concat( [normify(flow2), np.zeros(flow2.shape[:-1] + (1,))], axis=-1 ),
    ).export(MODEL_PATH.format(output="ners_flow_rgb", pose=pose_id, frame=fr0))

    # RGB 3D ners flow

    # flow3 = normify(dfm1 - dfm0)
    # trimesh.points.PointCloud(
    #   a_xyz0,
    #   colors=flow3,
    # ).export(MODEL_PATH.format(output="ners_flow3", pose=pose_id, frame=fr0))

    # flow3 = np.linalg.norm(dfm1 - dfm0, axis=-1, keepdims=True)
    # flow3 = np.log10(flow3 + 1)
    # flow3 = flow3 / np.max(flow3)
    # flow3 = np.repeat(flow3, 3, axis=-1)
    # trimesh.points.PointCloud(
    #   a_xyz0,
    #   colors=flow3,
    # ).export(MODEL_PATH.format(output="ners_flow3", pose=pose_id, frame=fr0))

    # HSV 2D unimatch flow
    unmflow_mag, unimflow_hue = flow_to_hue_flow(unimfl0)
    # print(unmflow_mag)

    trimesh.points.PointCloud(
      a_xyz0,
      colors=unimflow_hue,
    ).export(MODEL_PATH.format(output="unim_flow", pose=pose_id, frame=fr0))

  # RGB flow2
  if False:
    flow2_norm = normify(flow2)

    trimesh.points.PointCloud(
      a_xyz0,
      colors=np.concat([ flow2_norm, np.zeros( flow2_norm.shape[:-1] + (1,) ) ], axis=-1),
    ).export(MODEL_PATH.format(output="flow", pose=pose_id, frame=fr0))

def main():
  pose_ids = [
    # "221501007",

    # "220700191",
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

  for fr0 in tqdm(range(105, 413)):
  # for fr0 in tqdm(range(60,80)):
  # for fr0 in [1, 2, 3, 287, 288]:
  # for fr0 in [1]:
  # for fr0 in [287]:
    for pose_id in pose_ids:
      # print(fr0)
      make_flow(
        pose_id=pose_id,
        fr0=fr0,
      )

if __name__ == "__main__":
  main()
