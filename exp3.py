import itertools as itt

import torch
import torch.nn.functional as F
from ml_util import image_to_flow_hue, load_img
from PIL import Image
from tqdm import tqdm
from unimatch.unimatch import flow_warp
from util import osp


def main():
  inc = 24 / 73
  flow_mul = inc
  image_idx = 0
  image_acc = 0

  for flow_idx in tqdm(itt.count(1)):
    flow_path = f"/home/andrei/Downloads/happydepth/flow-{flow_idx}.png"
    if not osp.exists(flow_path):
      break

    flow = image_to_flow_hue(flow_path)
    flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
    flow = flow * flow_mul

    while image_acc < flow_idx:
      image_idx += 1
      image_acc += inc

      image_path = f"./nersemble-data/018/sequences/EMO-1-shout+laugh/images/cam_220700191.mp4:{image_idx}"

      image = load_img(image_path)
      image = image.permute(2, 0, 1).unsqueeze(0)
      image = F.interpolate(
        image,
        size=flow.shape[-2:],
        mode="bilinear",
        align_corners=True,
      )

      output_path = f"./unknown/warped-{image_idx}.png"

      image_warped = flow_warp(
        image.float(),
        flow.float(),
      )

      image_warped = image_warped.round().clip(0, 255).to(torch.uint8)
      image_warped = Image.fromarray(image_warped.squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy())

      image_warped.save(output_path)

if __name__ == "__main__":
  main()
