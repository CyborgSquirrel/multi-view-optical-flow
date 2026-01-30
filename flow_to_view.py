import functools as ft
import json
import os
import os.path as osp

import numpy as np
import torch
import trimesh
from dreifus.matrix import (CameraCoordinateConvention, Intrinsics, Pose,
                            PoseType)
from PIL import Image
from scipy.interpolate import griddata
from skimage.transform import resize  # pylint: disable
from tqdm import tqdm

from ml_util import (depth_to_image, flow_to_hue_flow, flow_to_image_hue,
                     image_to_depth, image_to_flow_hue, load_img, numpify)
from unimatch.unimatch import flow_warp

EXP_NAME = "warped-flow-2d-linear"

IMAGE_PATH = "./nersemble-data/018/sequences/EMO-1-shout+laugh/images/cam_{pose}.mp4:{frame}"
DEPTH_PATH = "./depth/NERS-9018_{pose}_depth-{frame}_checkpoint-300000.tiff"
BG_PATH = "./nersemble-data/018/sequences/BACKGROUND/image_{pose}.jpg"
FLOW_PATH = "./flow/{pose}-{frame}-flow.png"

MODEL_PATH = "./model/{output}_{pose}.ply"

WARPED_2D_FLOW_PATH = "./{exp}/{pose}-{frame}.png"

CAMERA_PARAMS_PATH = "/home/andrei/adl4cv-project/nersemble-data/018/calibration/camera_params.json"
@ft.cache
def camera_params():
  with open(CAMERA_PARAMS_PATH) as f:
    return json.load(f)

def make_flow(
  *,
  pose_id: str,
  pose_dst_id: str,
  fr0: int,
):
  fr1 = fr0 + 1

  fmt = dict(
    pose=pose_id,
  )

  output_path = WARPED_2D_FLOW_PATH.format(**fmt, frame=fr0, exp=EXP_NAME)
  os.makedirs(osp.dirname(output_path), exist_ok=True)
  if osp.exists(output_path):
    return

  # pose
  scale_factor = 9

  pose = Pose(
    camera_params()["world_2_cam"][pose_id],
    pose_type=PoseType.WORLD_2_CAM,
    camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
  )
  pose.change_pose_type(PoseType.CAM_2_WORLD)
  pose[:3, 3] *= scale_factor

  pose_dst = Pose(
    camera_params()["world_2_cam"][pose_dst_id],
    pose_type=PoseType.WORLD_2_CAM,
    camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
  )
  pose_dst.change_pose_type(PoseType.WORLD_2_CAM)
  pose_dst[:3, 3] *= scale_factor

  # intrinsics
  intr = Intrinsics(camera_params()["intrinsics"])

  # load images
  bg = load_img(BG_PATH.format(**fmt), output_type=np.ndarray)
  flow = image_to_flow_hue(FLOW_PATH.format(**fmt, frame=fr0))

  im0 = load_img(IMAGE_PATH.format(**fmt, frame=fr0), output_type=np.ndarray)
  im1 = load_img(IMAGE_PATH.format(**fmt, frame=fr1), output_type=np.ndarray)

  dt0 = image_to_depth(DEPTH_PATH.format(**fmt, frame=fr0))[..., None].astype(np.float64)
  # dt1 = image_to_depth(DEPTH_PATH.format(**fmt, frame=fr1))[..., None].astype(np.float64)

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
  # dt1  = dt1 .reshape(-1, dt1 .shape[-1])
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
    # dt1  = dt1 [~is_bg]
    rgb0 = rgb0[~is_bg]
    rgb1 = rgb1[~is_bg]

  if pose_id != pose_dst_id:
    # apply depth
    xy0h = np.concat([xy0, np.ones( xy0.shape[:-1] + (1,) )], axis=-1)
    xyz0 = xy0h * dt0

    xy1h = np.concat([xy1, np.ones( xy1.shape[:-1] + (1,) )], axis=-1)
    # xyz1 = xy1h * dt1
    xyz1 = xy1h * dt0

    # apply intrinsics
    xyz0 = np.einsum("ij,aj->ai", np.linalg.inv(intr), xyz0)
    xyz1 = np.einsum("ij,aj->ai", np.linalg.inv(intr), xyz1)

    # make homogeneous
    xyzh0 = np.concat([xyz0, np.ones( xyz0.shape[:-1] + (1,) )], axis=-1)
    xyzh1 = np.concat([xyz1, np.ones( xyz1.shape[:-1] + (1,) )], axis=-1)

    # apply pose
    xyzh0 = np.einsum("ij,aj->ai", pose, xyzh0)
    xyzh1 = np.einsum("ij,aj->ai", pose, xyzh1)
    # unapply dst pose
    xyzh0 = np.einsum("ij,aj->ai", pose_dst, xyzh0)
    xyzh1 = np.einsum("ij,aj->ai", pose_dst, xyzh1)

    xyz0 = xyzh0[..., :3] / xyzh0[..., [3]]
    xyz1 = xyzh1[..., :3] / xyzh1[..., [3]]

    # unapply intrinsics
    xyz0 = np.einsum("ij,aj->ai", intr, xyz0)
    xyz1 = np.einsum("ij,aj->ai", intr, xyz1)

    # sort by increasing z axis
    sort_idx = np.argsort(xyz1[..., 2])
    xyz0 = xyz0[sort_idx]
    xyz1 = xyz1[sort_idx]

    # unapply depth
    xy0 = xyz0[..., :2] / xyz0[..., [2]]
    xy1 = xyz1[..., :2] / xyz1[..., [2]]

  # get scene flow
  # if True:
  #   flow3 = xyz0 - xyz1

  #   # flow3_amax = np.max(np.abs(flow3))
  #   # flow3 = ((flow3 / flow3_amax) + 1) / 2

  flow2 = xy0 - xy1



  ### warp with interpolation
  dt_dst0 = image_to_depth(DEPTH_PATH.format(pose=pose_dst_id, frame=fr0))[..., None].astype(np.float64)
  dt_dst0 = dt_dst0.reshape(-1, dt_dst0.shape[-1])

  xyh_dst0 = np.concat([xy, np.ones( xy.shape[:-1] + (1,) )], axis=-1)
  xyz_dst0 = xyh_dst0 * dt_dst0
  flow2_interp = griddata(xyz0, flow2, xyz_dst0, fill_value=0, method="linear")
  # flow2_interp = griddata(xyz0, flow2, xyz_dst0, fill_value=0, method="nearest")
  flow_img = np.zeros(shape + (2,))
  idk = xy // 4

  flow_img = np.zeros(shape + (2,))
  flow_img[idk[...,1], idk[...,0]] = flow2_interp 
  flow_img = flow_to_image_hue(flow_img)
  flow_img.save(output_path, exif=flow_img.getexif())



  ### warp with binning
  # flow_img = np.zeros(shape + (2,))
  # xym = np.round(xy1 / mul).astype(int)
  # mask = (
  #     (xym[...,0] >= 0) & (xym[...,0] < shape[1])
  #   & (xym[...,1] >= 0) & (xym[...,1] < shape[0])
  # )
  # flow2 = flow2[mask]
  # xym = xym[mask]
  # flow_img[xym[...,1], xym[...,0]] = flow2
  # flow_img = flow_to_image_hue(flow_img)
  # flow_img.save(output_path, exif=flow_img.getexif())



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



  # trimesh.points.PointCloud(
  #   xyz1,
  #   colors=flow3,
  # ).export(MODEL_PATH.format(output="flow", pose=pose_id))

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

  pose_dst_id = "221501007"

  for fr0 in tqdm(range(1, 413+1)):
    for pose_id in pose_ids:
      make_flow(
        pose_id=pose_id,
        pose_dst_id=pose_dst_id,
        fr0=fr0,
      )

if __name__ == "__main__":
  main()
