# optical flows are in the warped-flow-2d-norm folder.
# the flows are in .png format. For each camera, the flows are named xxxxx-1.png with 1 being frame index.
# calulcate error between 221501007 and all other 15 camera optical flows for each frame
# use masks to ignore white regions in the images in projected (other) cameras

import os
import cv2 as cv
import numpy as np
from ml_util import image_to_flow_hue
import subprocess

def calculate_diff(flow1, flow2, mask):
	# Calculate Mean Absolute Error between two flows with a mask
	diff = np.sum(np.abs(flow1 - flow2), axis=2)  # shape (H, W)
	masked_diff = diff * mask
	return masked_diff

def make_video(cam):
	subprocess.call(["ffmpeg", "-framerate", "73", "-i", f"output/{cam}/%d.png", "-vf", "crop=iw-mod(iw\\,2):ih-mod(ih\\,2)", "-c:v", "libx264", "-pix_fmt", "yuv420p", f"output/{cam}/output.mp4"])

def main():
	flow_dir = '../warped-flow-2d-norm/'
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
	max_diff = {other_camera: 0.0 for other_camera in other_cameras}
	average_diff = 0.0
	num_frames = 412
	os.makedirs('output', exist_ok=True)
	for cam in other_cameras:
		os.makedirs(os.path.join('output', f'{cam}'), exist_ok=True)
	os.makedirs(os.path.join('output', f'average_of_15'), exist_ok=True)
	
	for i in range(num_frames):
		frame_idx = i + 1
		target_flow_path = os.path.join(flow_dir, f'{target_camera}-{frame_idx}.png')
		target_flow = image_to_flow_hue(target_flow_path) # shape (H, W, 2)
		total_flow = np.zeros(target_flow.shape, dtype=target_flow.dtype)
		total_mask = np.zeros(target_flow.shape[:2], dtype=int)

		for other_camera in other_cameras:
			other_flow_path = os.path.join(flow_dir, f'{other_camera}-{frame_idx}.png')
			other_flow = image_to_flow_hue(other_flow_path) # shape (H, W, 2)
			# Create mask from target flow
			mask = np.linalg.norm(other_flow, axis=2) > 0.1  # Threshold to create mask
			total_flow = other_flow * mask[:, :, np.newaxis]
			total_mask += mask.astype(int)
			diff = calculate_diff(target_flow, other_flow, mask)
			max_diff[other_camera] = max(max_diff[other_camera], np.max(diff))
		copy_mask = total_mask.copy()
		copy_mask[copy_mask == 0] = 1 # To avoid division by zero
		copy_mask = copy_mask[:, :, np.newaxis]
		average_flow = total_flow / copy_mask
		diff = calculate_diff(target_flow, average_flow, total_mask > 0)
		average_diff = max(average_diff, np.max(diff))


	for i in range(num_frames):
		frame_idx = i + 1
		target_flow_path = os.path.join(flow_dir, f'{target_camera}-{frame_idx}.png')
		target_flow = image_to_flow_hue(target_flow_path) # shape (H, W, 2)
		total_flow = np.zeros(target_flow.shape, dtype=target_flow.dtype)
		total_mask = np.zeros(target_flow.shape[:2], dtype=int)
		for other_camera in other_cameras:
			other_flow_path = os.path.join(flow_dir, f'{other_camera}-{frame_idx}.png')
			other_flow = image_to_flow_hue(other_flow_path) # shape (H, W, 2)
			# Create mask from target flow
			mask = np.linalg.norm(other_flow, axis=2) > 0.1  # Threshold to create mask
			total_flow += other_flow * mask[:, :, np.newaxis]
			total_mask += mask.astype(int)
			diff = calculate_diff(target_flow, other_flow, mask)
			diff = (diff / max_diff[other_camera]) * 255.0
			print(f'Frame {frame_idx}, Cameras {target_camera} to {other_camera}, Mean Absolute Error: {np.mean(diff)}')
			cv.imwrite(os.path.join('output', f'{other_camera}', f'{frame_idx}.png'), diff.astype(np.uint8))
		copy_mask = total_mask.copy()
		copy_mask[copy_mask == 0] = 1 # To avoid division by zero
		copy_mask = copy_mask[:, :, np.newaxis]
		average_flow = total_flow / copy_mask
		diff = calculate_diff(target_flow, average_flow, total_mask > 0)
		diff = (diff / average_diff) * 255.0
		print(f'Frame {frame_idx}, Average of 15, Mean Absolute Error: {np.mean(diff)}')
		cv.imwrite(os.path.join('output', f'average_of_15', f'{frame_idx}.png'), ((diff / np.max(diff)) * 255.0).astype(np.uint8))
			
	for other_camera in other_cameras:
		make_video(other_camera)
		print(f"Video for camera {other_camera} created.")
	make_video('average_of_15')
	print("Video for average of 15 cameras created.")
	for diff in max_diff.items():
		print(f'Max difference for camera {diff[0]}: {diff[1]}')
	return


if __name__ == "__main__":
	main()