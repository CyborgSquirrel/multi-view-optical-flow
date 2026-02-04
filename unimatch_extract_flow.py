import contextlib as ctl
import itertools as itt
import os
import os.path as osp
import queue
import threading
import time
from collections import deque as Deque
from datetime import datetime, timezone
from queue import Queue

import imageio as iio
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ml_util import (flow_to_image_hue, unimatch_get_transform,
                     unimatch_load_img)
from unimatch.unimatch import UniMatch, flow_warp
from util import Oneshot, ThreadPool, resolve_local_path

# Set this because otherwise PyTorch hangs forever sometimes
# os.environ["OMP_NUM_THREADS"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NOW = datetime.now(timezone.utc).isoformat()

BASE_DIR = osp.join("experiments", f"exp2-{NOW}")
os.makedirs(BASE_DIR, exist_ok=True)

CHECKPOINT = "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"

FRAME_GAP = 0
# INPUT_SCALE = 0.5
INPUT_SCALE = 0.25
# INPUT_SCALE = 0.1

PADDING_FACTOR         = 32
NUM_SCALES             = 2
FEATURE_CHANNELS       = 128
UPSAMPLE_FACTOR        = 4
NUM_HEAD               = 1
FFN_DIM_EXPANSION      = 4
NUM_TRANSFORMER_LAYERS = 6
REG_REFINE             = True

ATTN_TYPE        = "swin"
ATTN_SPLITS_LIST = [2, 8]
CORR_RADIUS_LIST = [-1, 4]
PROP_RADIUS_LIST = [-1, 1]
NUM_REG_REFINE   = 6

inference_queue = Queue()

def thread_inference():
  batch_size = 2
  # batch_size = 1

  # Initialize model
  model = UniMatch(
    feature_channels=FEATURE_CHANNELS,
    num_scales=NUM_SCALES,
    upsample_factor=UPSAMPLE_FACTOR,
    num_head=NUM_HEAD,
    ffn_dim_expansion=FFN_DIM_EXPANSION,
    num_transformer_layers=NUM_TRANSFORMER_LAYERS,
    reg_refine=REG_REFINE,
    task="flow",
  ).to(device)
  model.eval()

  # Load checkpoint
  checkpoint = resolve_local_path(CHECKPOINT)
  checkpoint = torch.load(checkpoint, map_location=device)
  model.load_state_dict(checkpoint["model"])

  running = True
  while running:
    tasks = []

    try:
      while len(tasks) <= 0:
        time.sleep(1)
        for _ in range(batch_size):
          task = inference_queue.get_nowait()
          if task is None:
            running = False
            break
          tasks.append(task)
    except queue.Empty:
      if len(tasks) <= 0:
        continue

    img0 = torch.stack([task[1] for task in tasks]).to(device)
    img1 = torch.stack([task[2] for task in tasks]).to(device)

    with torch.no_grad():
      results_dict = model(
        img0,
        img1,
        attn_type=ATTN_TYPE,
        attn_splits_list=ATTN_SPLITS_LIST,
        corr_radius_list=CORR_RADIUS_LIST,
        prop_radius_list=PROP_RADIUS_LIST,
        num_reg_refine=NUM_REG_REFINE,
        task="flow",
      )
    flow = results_dict["flow_preds"]
    del results_dict
    flow = flow[0].cpu()  # [B, 2, H, W]

    for i, task in enumerate(tasks):
      chan = task[0]
      chan.put(flow[i])

def main():
  def thread_frame(fr0, img0, fr1, img1):
    t0 = time.perf_counter_ns()

    img0 = unimatch_load_img(img0)
    img1 = unimatch_load_img(img1)

    img0 = F.interpolate(
      img0.unsqueeze(0),
      size=tuple(round(a*INPUT_SCALE) for a in img0.shape[-2:]),
      mode="bilinear",
      align_corners=True,
    ).squeeze(0)
    img1 = F.interpolate(
      img1.unsqueeze(0),
      size=tuple(round(a*INPUT_SCALE) for a in img1.shape[-2:]),
      mode="bilinear",
      align_corners=True,
    ).squeeze(0)

    trans_fn, restore_fn = unimatch_get_transform(
      [img0, img1],
      padding_factor=PADDING_FACTOR,
    )

    img0i = trans_fn(img0)
    img1i = trans_fn(img1)

    # Estimate backward flow
    resp = Oneshot()
    inference_queue.put((resp, img1i, img0i))
    flow = resp.get()
    flow = flow.squeeze(0)  # [2, H, W]
    flow = restore_fn(flow, scale_flow=True)

    img1w = flow_warp(img0.unsqueeze(0), flow.unsqueeze(0)).squeeze(0)

    # Prepare for writing outputs
    # frames = ",".join([str(fr0), str(fr1)])
    frames = str(fr0)

    flow = flow.permute(1, 2, 0)
    img1w = img1w.permute(1, 2, 0)
    img1 = img1.permute(1, 2, 0)

    # err = (img1w - img1) ** 2
    err = torch.abs(img1w - img1)
    err = torch.mean(err, dim=-1)
    err = torch.clip(err, 0, 255)

    # Convert to images
    flow = flow_to_image_hue(flow)
    img1w = Image.fromarray(img1w.to(torch.uint8).numpy())
    err = Image.fromarray(err.to(torch.uint8).numpy())

    flow.save(osp.join(BASE_DIR, f"{frames}-flow.png"), exif=flow.getexif())
    img1w.save(osp.join(BASE_DIR, f"{frames}-img1w.png"))
    err.save(osp.join(BASE_DIR, f"{frames}-err.png"))

    t1 = time.perf_counter_ns()
    delta_time = t1 - t0
    delta_time = delta_time / 10**9  # convert to secs

  with ctl.ExitStack() as stack:
    t = threading.Thread(target=thread_inference)
    t.start()
    stack.callback(lambda: inference_queue.put(None))
    stack.callback(lambda: t.join())

    path = "./nersemble/018/sequences/EMO-1-shout+laugh/images/cam_220700191.mp4"

    # Get number of frames
    frames_no = None
    with iio.imopen(path, "r", plugin="pyav") as f:
      prop = f.properties()
      frames_no = prop.n_images

    # Load video
    frame_iter = iio.imiter(path, plugin="pyav")
    frame_iter = enumerate(frame_iter)
    frame_iter = tqdm(frame_iter, total=frames_no)

    # Set up frame deque
    maxlen = FRAME_GAP + 2
    dq = Deque(maxlen=maxlen)

    # pool: ThreadPool = stack.enter_context(ThreadPool(max_workers=32))
    pool: ThreadPool = stack.enter_context(ThreadPool(max_workers=4))

    for frame in frame_iter:
      dq.append(frame)
      if len(dq) < maxlen:
        continue

      fr0, img0 = dq[ 0]
      fr1, img1 = dq[-1]

      pool.submit(thread_frame, fr0, img0, fr1, img1)

if __name__ == "__main__":
  main()
