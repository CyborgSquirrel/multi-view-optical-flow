"""Combine flows into a single flow"""

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

CAMERA_PARAMS_PATH = "/home/andrei/adl4cv-project/nersemble-data/018/calibration/camera_params.json"

def main():
	# camera params
	with open(CAMERA_PARAMS_PATH) as f:
		camera_params = json.load(f)

	poses = [
		"220700191",
		"221501007",
		"222200036",
		"222200037",
		"222200038",
		# "222200039",
		# "222200040",
		# "222200041",
		# "222200042",
	]

	for pose_id in tqdm(poses):
		fr0 = 100
		fr1 = fr0+1

		args = dict(
			pose = pose_id,
		)

		# pose
		pose = Pose(
		  camera_params["world_2_cam"][pose_id],
		  pose_type=PoseType.WORLD_2_CAM,
		  camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
		)
		pose.change_pose_type(PoseType.CAM_2_WORLD)
		# print(pose)

		scale_factor = 9
		pose[:3, 3] *= scale_factor

		# intrinsics
		intr = Intrinsics(camera_params["intrinsics"])

		# load images
		bg = load_img(BG_PATH_FMT.format(**args), output_type=np.ndarray)
		flow = image_to_flow_hue(FLOW_PATH_FMT.format(**args, frame=fr0))

		im0 = load_img(IMAGE_PATH_FMT.format(**args, frame=fr0), output_type=np.ndarray)
		im1 = load_img(IMAGE_PATH_FMT.format(**args, frame=fr1), output_type=np.ndarray)

		dt0 = image_to_depth(DEPTH_PATH_FMT.format(**args, frame=fr0))
		dt1 = image_to_depth(DEPTH_PATH_FMT.format(**args, frame=fr1))

		# resize images
		shape = dt0.shape[:2]

		im0 = resize(im0, shape, preserve_range=True)
		im1 = resize(im1, shape, preserve_range=True)
		bg = resize(bg, shape, preserve_range=True)

		# do the thing
		h, w = shape
		mul = 4
		x, y = np.meshgrid(np.arange(w) * mul, np.arange(h) * mul)

		xy = np.stack([x, y], axis=-1)

		# rgb
		rgb0 = im0 / 255
		rgb1 = im1 / 255
		bg = bg / 255

		# flow
		xy0 = xy + flow * mul
		xy1 = xy

		# apply depth
		# TODO(?): warp dt0 using flow_warp
		xy0h = np.concat([xy0, np.ones( xy0.shape[:-1] + (1,) )], axis=-1)
		xyz0 = xy0h * dt0[..., None]

		xy1h = np.concat([xy1, np.ones( xy1.shape[:-1] + (1,) )], axis=-1)
		# xyz1 = xy1h * dt1[..., None]
		xyz1 = xy1h * dt0[..., None]

		# flatten
		xyz0 = xyz0.reshape(-1, xyz0.shape[-1])
		xyz1 = xyz1.reshape(-1, xyz1.shape[-1])
		rgb0 = rgb0.reshape(-1, rgb0.shape[-1])
		rgb1 = rgb1.reshape(-1, rgb1.shape[-1])
		bg = bg.reshape(-1, bg.shape[-1])

		# remove background
		is_bg = (
		    (np.linalg.norm(rgb0 - bg, axis=-1, ord=2) <= 0.2)
		  | (np.linalg.norm(rgb1 - bg, axis=-1, ord=2) <= 0.2)
		)
		xyz0 = xyz0[~is_bg]
		xyz1 = xyz1[~is_bg]
		rgb0 = rgb0[~is_bg]
		rgb1 = rgb1[~is_bg]

		# print(np.min(flow), np.max(flow))
		# print(np.min(dt0), np.max(dt0))
		# print(np.min(dt1), np.max(dt1))

		# apply intrinsics
		xyz0 = np.einsum("ij,aj->ai", np.linalg.inv(intr), xyz0)
		xyz1 = np.einsum("ij,aj->ai", np.linalg.inv(intr), xyz1)

		# apply pose
		xyz0h = np.concat([xyz0, np.ones( xyz0.shape[:-1] + (1,) )], axis=-1)
		xyz0h = np.einsum("ij,aj->ai", pose, xyz0h)
		xyz0 = xyz0h[..., :3] / xyz0h[..., [3]]

		xyz1h = np.concat([xyz1, np.ones( xyz1.shape[:-1] + (1,) )], axis=-1)
		xyz1h = np.einsum("ij,aj->ai", pose, xyz1h)
		xyz1 = xyz1h[..., :3] / xyz1h[..., [3]]

		trimesh.points.PointCloud(
		  xyz1,
		  colors=rgb1,
		).export(f"color_{pose_id}.ply")

		# get scene flow
		if True:
			flow3 = xyz0 - xyz1

			flow3_amax = np.max(np.abs(flow3))
			flow3 = ((flow3 / flow3_amax) + 1) / 2

		xyz = xyz1
		rgb = flow3
		# rgb = np.repeat(flow_mag[..., None], 3, axis=-1)
		# rgb = (
		# 	np.repeat(np.array(rgb)[None, :], len(xyz), axis=0)
		# )

		trimesh.points.PointCloud(
		  xyz,
		  colors=rgb,
		).export(f"flow_{pose_id}.ply")

if __name__ == "__main__":
	main()
