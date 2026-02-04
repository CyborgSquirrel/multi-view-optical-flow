# get the average flow from multiple interpolated flows

import os
import cv2 as cv
import numpy as np
from ml_util import flow_to_image_hue, image_to_flow_hue
from scipy.interpolate import griddata
import json

CAMERA_PARAMS_PATH = "/home/andrei/adl4cv-project/nersemble-data/018/calibration/camera_params.json"
def camera_params():
  with open(CAMERA_PARAMS_PATH) as f:
    return json.load(f)

def average_flows(flow_list, mask_list):
	# Compute the average flow considering only valid pixels from the masks
	sum_flow = np.zeros(flow_list[0].shape, dtype=flow_list[0].dtype)
	sum_mask = np.zeros(flow_list[0].shape[:2], dtype=int)

	for flow, mask in zip(flow_list, mask_list):
		sum_flow += flow * mask[:, :, np.newaxis]
		sum_mask += mask.astype(int)

	# Avoid division by zero
	avg_flow = np.zeros_like(sum_flow)
	valid_pixels = sum_mask > 0
	avg_flow[valid_pixels] = sum_flow[valid_pixels] / sum_mask[valid_pixels, np.newaxis]

	return avg_flow

def save_average_flows(flow_dir, target_camera, other_cameras, num_frames=412, output_dir='average_flows'):
	os.makedirs(output_dir, exist_ok=True)

	for i in range(num_frames):
		frame_idx = i + 1
		flow_list = []
		mask_list = []

		for other_camera in other_cameras:
			other_flow_path = os.path.join(flow_dir, f'{other_camera}-{frame_idx}.png')
			other_flow = image_to_flow_hue(other_flow_path) # shape (H, W, 2)
			mask = np.linalg.norm(other_flow, axis=2) > 0.1  # Threshold to create mask
			flow_list.append(other_flow)
			mask_list.append(mask.astype(int))
		target_camera_path = os.path.join(flow_dir, f'{target_camera}-{frame_idx}.png')
		target_flow = image_to_flow_hue(target_camera_path) # shape (H, W, 2)
		main_mask = np.linalg.norm(target_flow, axis=2) > 0.1
		average_flow = average_flows(flow_list, mask_list)
		average_i_flow = interpolate(average_flow, main_mask, method='linear')

		# Save the average flow as an image
		avg_flow_image = flow_to_image_hue(average_flow)
		avg_i_flow_image = flow_to_image_hue(average_i_flow)
		avg_flow_image.save(os.path.join(output_dir, f'average_{frame_idx}.png'), exif=avg_flow_image.getexif())
		avg_i_flow_image.save(os.path.join(output_dir, f'average_interpolated_{frame_idx}.png'), exif=avg_i_flow_image.getexif())
		print(f'Processed frame {frame_idx} for average flow.')


def interpolate(flow, mask, method='linear'):
	# Interpolate missing flow values using griddata
	h, w, _ = flow.shape
	xx, yy = np.meshgrid(np.arange(w), np.arange(h))

	valid_points = np.nonzero(mask)
	invalid_points = np.nonzero(~mask)

	# Interpolate for both flow components
	flow_x = griddata(
		(valid_points),
		flow[valid_points][:, 0],
		(xx[invalid_points], yy[invalid_points]),
		method=method,
		fill_value=0
	)
	flow_y = griddata(
		(valid_points[1], valid_points[0]),
		flow[valid_points][:, 1],
		(xx[invalid_points], yy[invalid_points]),
		method=method,
		fill_value=0
	)

	interpolated_flow = flow.copy()
	interpolated_flow[invalid_points] = np.stack((flow_x, flow_y), axis=-1)

	return interpolated_flow
    


def main():
	flow_dir = '../warped-flow-2d-norm'
	target_camera = '221501007'
	other_cameras = [
	'220700191', 
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

	save_average_flows(flow_dir, target_camera, other_cameras)

if __name__ == '__main__':
	main()

