import json

import numpy as np
import torch
import trimesh
from dreifus.matrix import (CameraCoordinateConvention, Intrinsics, Pose,
                            PoseType)
from PIL import Image
from skimage.transform import resize  # pylint: disable
from tqdm import tqdm

from ml_util import image_to_depth, image_to_flow_hue, load_img
from unimatch.unimatch import flow_warp

# poses:
# 220700191 221501007 222200036 222200037 222200038 222200039 222200040 222200041 222200042 222200043 222200044 222200045 222200046 222200047 222200048 222200049

IMAGE_PATH_FMT = "./nersemble-data/018/sequences/EMO-1-shout+laugh/images/cam_{pose}.mp4:{frame}"
# DEPTH_PATH_FMT = "/media/andrei/gdrive/adl4cv/nersemble-out/NERS-9018_{pose}_depth-{frame}_checkpoint-300000.png"
DEPTH_PATH_FMT = "./depth/NERS-9018_{pose}_depth-{frame}_checkpoint-300000.tiff"
BG_PATH_FMT = "./nersemble-data/018/sequences/BACKGROUND/image_{pose}.jpg"
FLOW_PATH_FMT = "./flow/cam_{pose}/{frame}-flow.png"

MODEL_PATH_FMT = "./model/{output}_{pose}.ply"

CAMERA_PARAMS_PATH = "/home/andrei/adl4cv-project/nersemble-data/018/calibration/camera_params.json"

def main():
  # camera params
  with open(CAMERA_PARAMS_PATH) as f:
    camera_params = json.load(f)

  pose_ids = [
    "220700191",
    "221501007",
    "222200036",
    "222200037",
    "222200038",
    "222200039",
    "222200040",
    "222200041",
    # "222200042",
  ]

  pose_dst_id = "221501007"

  fr = 100

  for pose_id in tqdm(pose_ids):
    # pose
    scale_factor = 9

    pose = Pose(
      camera_params["world_2_cam"][pose_id],
      pose_type=PoseType.WORLD_2_CAM,
      camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
    )
    pose.change_pose_type(PoseType.CAM_2_WORLD)
    pose[:3, 3] *= scale_factor

    # pose_inv = pose.copy()
    # pose_inv.change_pose_type(PoseType.WORLD_2_CAM)

    # print(np.matmul(pose, pose_inv).numpy())

    # break

    pose_dst = Pose(
      camera_params["world_2_cam"][pose_dst_id],
      pose_type=PoseType.WORLD_2_CAM,
      camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
    )
    pose_dst.change_pose_type(PoseType.WORLD_2_CAM)
    pose_dst[:3, 3] *= scale_factor

    # intrinsics
    intr = Intrinsics(camera_params["intrinsics"])

    # load images
    args = dict(
      pose=pose_id,
      frame=fr,
    )

    bg = load_img(BG_PATH_FMT.format(**args), output_type=np.ndarray)
    im = load_img(IMAGE_PATH_FMT.format(**args), output_type=np.ndarray)
    dt = image_to_depth(DEPTH_PATH_FMT.format(**args))

    # resize images
    shape = dt.shape[:2]

    im = resize(im, shape, preserve_range=True)
    bg = resize(bg, shape, preserve_range=True)

    # do the thing
    h, w = shape
    mul = 4
    x, y = np.meshgrid(np.arange(w) * mul, np.arange(h) * mul)

    xy = np.stack([x, y], axis=-1)

    # rgb
    rgb = im / 255
    bg = bg / 255

    # apply depth
    xyh = np.concat([xy, np.ones( xy.shape[:-1] + (1,) )], axis=-1)
    xyz = xyh * dt[..., None]

    # flatten
    xyz = xyz.reshape(-1, xyz.shape[-1])
    rgb = rgb.reshape(-1, rgb.shape[-1])
    bg = bg.reshape(-1, bg.shape[-1])

    # remove background
    is_bg = (np.linalg.norm(rgb - bg, axis=-1, ord=2) <= 0.2)
    xyz = xyz[~is_bg]
    rgb = rgb[~is_bg]

    # apply intrinsics
    xyz = np.einsum("ij,aj->ai", np.linalg.inv(intr), xyz)

    if pose_id != pose_dst_id:
      xyzh = np.concat([xyz, np.ones( xyz.shape[:-1] + (1,) )], axis=-1)

      # apply pose
      xyzh = np.einsum("ij,aj->ai", pose, xyzh)
      # unapply dst pose
      xyzh = np.einsum("ij,aj->ai", pose_dst, xyzh)

      xyz = xyzh[..., :3] / xyzh[..., [3]]

    trimesh.points.PointCloud(
      xyz,
      colors=rgb,
    ).export(MODEL_PATH_FMT.format(output="color", pose=pose_id))

if __name__ == "__main__":
  main()
