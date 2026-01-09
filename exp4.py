import json

import numpy as np
import torch
import trimesh
from dreifus.matrix import (CameraCoordinateConvention, Intrinsics, Pose,
                            PoseType)
from PIL import Image
from skimage.transform import resize  # pylint: disable
from tqdm import tqdm

from ml_util import image_to_depth, load_img

# poses:
# 220700191 221501007 222200036 222200037 222200038 222200039 222200040 222200041 222200042 222200043 222200044 222200045 222200046 222200047 222200048 222200049

FRAME_PATH_FMT = "./nersemble-data/018/sequences/EMO-1-shout+laugh/images/cam_{pose}.mp4:{frame}"
# DEPTH_PATH_FMT = "/media/andrei/gdrive/adl4cv/nersemble-out/NERS-9018_{pose}_depth-{frame}_checkpoint-300000.png"
DEPTH_PATH_FMT = "./depth/NERS-9018_{pose}_depth-{frame}_checkpoint-300000.png"
BG_PATH_FMT = "./nersemble-data/018/sequences/BACKGROUND/image_{pose}.jpg"

CAMERA_PARAMS_PATH = "/home/andrei/adl4cv-project/nersemble-data/018/calibration/camera_params.json"

def main():
	# camera params
	with open(CAMERA_PARAMS_PATH) as f:
		camera_params = json.load(f)

	xyz_all = []
	rgb_all = []

	poses = [
		"220700191",
		"221501007",
		"222200036",
		"222200037",
		"222200038",
		"222200039",
		"222200040",
		"222200041",
		"222200042",
	]

	for pose in tqdm(poses):
		args = dict(
			pose = pose,
			frame = 100,
		)
		frame_path = FRAME_PATH_FMT.format(**args)
		depth_path = DEPTH_PATH_FMT.format(**args)
		bg_path = BG_PATH_FMT.format(**args)

		# pose
		pose = Pose(
		  camera_params["world_2_cam"][pose],
		  pose_type=PoseType.WORLD_2_CAM,
		  camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
		)
		pose.change_pose_type(PoseType.CAM_2_WORLD)
		# pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL)
		pose.swap_axes(["x", "-z", "y"])
		print(pose)

		scale_factor = 9
		pose[:3, 3] *= scale_factor

		# intrinsics
		intr = Intrinsics(camera_params["intrinsics"])

		# load images
		frame = load_img(frame_path, output_type=np.ndarray)
		depth = image_to_depth(depth_path)
		bg = load_img(bg_path, output_type=np.ndarray)

		frame = resize(frame, depth.shape[:2], preserve_range=True)
		frame = np.clip(np.round(frame), 0, 255).astype(np.uint8)

		bg = resize(bg, depth.shape[:2], preserve_range=True)
		bg = np.clip(np.round(bg), 0, 255).astype(np.uint8)

		# Image.fromarray(frame).show()
		# Image.fromarray(depth).show()

		# do the thing
		h, w = depth.shape[:2]
		mul = 4
		# mul = 1
		x, y = np.meshgrid(np.arange(w) * mul, np.arange(h) * mul)

		xy = np.stack([x, y], axis=-1)

		xy = np.concat([xy, np.ones( xy.shape[:-1] + (1,) )], axis=-1)

		# rgb
		rgb = frame / 255
		bg = bg / 255

		# apply depth
		z = depth
		xyz = xy * z[..., None]

		# flatten
		xyz = xyz.reshape(-1, xyz.shape[-1])
		rgb = rgb.reshape(-1, rgb.shape[-1])
		bg = bg.reshape(-1, bg.shape[-1])

		# remove background
		mask = np.linalg.norm(rgb - bg, axis=-1, ord=2) > 0.2
		xyz = xyz[mask]
		rgb = rgb[mask]

		# apply intrinsics
		xyz = np.einsum("ij,aj->ai", np.linalg.inv(intr), xyz)

		# xyz_all.append(xyz)
		# rgb_all.append(rgb * 0.5)

		# xyz = generate_sphere_points(1000, 10)

		# apply pose
		xyzh = np.concat([xyz, np.ones( xyz.shape[:-1] + (1,) )], axis=-1)
		xyzh = np.einsum("ij,aj->ai", pose, xyzh)
		xyz = xyzh[..., :3] / xyzh[..., [3]]

		# filter out bg
		# bg_dist = np.linalg.norm(rgb - bg_rgb[None, :], ord=2, axis=-1)
		# mask = bg_dist <= 0.8
		# xyz = xyz[mask]
		# rgb = rgb[mask]

		xyz_all.append(xyz)
		rgb_all.append(rgb)
		# rgb_all.append(
		# 	np.repeat(np.array(rgb)[None, :], len(xyz), axis=0)
		# )

	xyz = np.concat(xyz_all)
	rgb = np.concat(rgb_all)

	pc = trimesh.points.PointCloud(
	  xyz,
	  colors=rgb,
	)
	pc.export("frame0_debug.ply")

if __name__ == "__main__":
	main()
