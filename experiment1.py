import contextlib as ctl
import csv
import itertools as itt
import json
import os
import queue
import threading
import time
from collections import deque as Deque
from queue import Queue

import imageio as iio
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ml_util import unimatch_get_transform, unimatch_load_img
from unimatch.unimatch import UniMatch, flow_warp
from util import Oneshot, ThreadPool, resolve_local_path

# Set this because otherwise PyTorch hangs forever sometimes
os.environ['OMP_NUM_THREADS'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NOW = int(time.time())

CSV_PATH = f'experiments/experiment1-{NOW}.csv'
META_PATH = f'experiments/experiment1-{NOW}.json'

CHECKPOINT = 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth'
FRAME_GAP = 0
INFER_SIZE = None
# INFER_SIZE = (128, 128)
# INFER_SCALE = 0.1
INFER_SCALE = 0.05
# INFER_SCALE = 0.025

PADDING_FACTOR         = 32
NUM_SCALES             = 2
FEATURE_CHANNELS       = 128
UPSAMPLE_FACTOR        = 4
NUM_HEAD               = 1
FFN_DIM_EXPANSION      = 4
NUM_TRANSFORMER_LAYERS = 6
REG_REFINE             = True

ATTN_TYPE        = 'swin'
ATTN_SPLITS_LIST = [2, 8]
CORR_RADIUS_LIST = [-1, 4]
PROP_RADIUS_LIST = [-1, 1]
NUM_REG_REFINE   = 6

inference_queue = Queue()

def thread_inference():
  batch_size = 16

  # Initialize model
  model = UniMatch(
    feature_channels=FEATURE_CHANNELS,
    num_scales=NUM_SCALES,
    upsample_factor=UPSAMPLE_FACTOR,
    num_head=NUM_HEAD,
    ffn_dim_expansion=FFN_DIM_EXPANSION,
    num_transformer_layers=NUM_TRANSFORMER_LAYERS,
    reg_refine=REG_REFINE,
    task='flow',
  ).to(device)
  model.eval()

  # Load checkpoint
  checkpoint = resolve_local_path(CHECKPOINT)
  checkpoint = torch.load(checkpoint, map_location=device)
  model.load_state_dict(checkpoint['model'])

  running = True
  while running:
    tasks = []

    time.sleep(1)
    try:
      for _ in range(batch_size):
        task = inference_queue.get_nowait()
        if task is None:
          running = False
          break
        tasks.append(task)
    except queue.Empty:
      pass

    img0 = torch.stack([task[1] for task in tasks])
    img1 = torch.stack([task[2] for task in tasks])

    with torch.no_grad():
      results_dict = model(
        img0, img1,
        attn_type=ATTN_TYPE,
        attn_splits_list=ATTN_SPLITS_LIST,
        corr_radius_list=CORR_RADIUS_LIST,
        prop_radius_list=PROP_RADIUS_LIST,
        num_reg_refine=NUM_REG_REFINE,
        task='flow',
      )
    flow = results_dict['flow_preds']
    flow = flow[0]  # [B, 2, H, W]

    for i, task in enumerate(tasks):
      chan = task[0]
      chan.put(flow[i])

def main():
  def thread_frame(fr0, img0, fr1, img1):
    t0 = time.perf_counter_ns()

    img0 = unimatch_load_img(img0)
    img1 = unimatch_load_img(img1)

    trans_fn, restore_fn = unimatch_get_transform(
      [img0, img1],
      infer_size=INFER_SIZE,
      infer_scale=INFER_SCALE,
      padding_factor=PADDING_FACTOR,
    )

    img0_infer = trans_fn(img0)
    img1_infer = trans_fn(img1)

    # Image.fromarray(img0.permute(1, 2, 0).cpu().numpy().astype(np.uint8)).show("img0")
    # Image.fromarray(img1.permute(1, 2, 0).cpu().numpy().astype(np.uint8)).show("img1")

    # Estimate forward flow
    resp = Oneshot()
    inference_queue.put((resp, img0_infer, img1_infer))
    flow = resp.get()

    flow = flow.squeeze(0)  # [2, H, W]
    flow = restore_fn(flow, scale_flow=True)
    flow_fwd = flow

    # Forward warp
    img1w = flow_warp(img0.unsqueeze(0), flow.unsqueeze(0)).squeeze(0)
    img1w_infer = trans_fn(img1w)
    # Image.fromarray(img1w.permute(1, 2, 0).cpu().numpy().astype(np.uint8)).show("img1w")

    # Estimate backward flow from warped frame
    resp = Oneshot()
    inference_queue.put((resp, img1w_infer, img0_infer))
    flow = resp.get()

    flow = flow.squeeze(0)  # [2, H, W]
    flow = restore_fn(flow, scale_flow=True)
    flow_bwd = flow

    # Forward flow should be opposite of backward flow
    loss_mae = F.l1_loss(flow_fwd, -flow_bwd).item()
    loss_mse = F.mse_loss(flow_fwd, -flow_bwd).item()

    t1 = time.perf_counter_ns()
    delta_time = t1 - t0
    delta_time = delta_time / 10**9  # convert to secs

    # Save result
    with writer_lock:
      writer.writerow(dict(
        video=path,
        frames=json.dumps([fr0, fr1]),
        loss_mae=loss_mae,
        loss_mse=loss_mse,
        delta_time=delta_time,
      ))
      f.flush()

  with ctl.ExitStack() as stack:
    t = threading.Thread(target=thread_inference)
    t.start()
    stack.callback(lambda: inference_queue.put(None))
    stack.callback(lambda: t.join())

    # Dump meta
    with open(META_PATH, 'w') as f:
      meta = dict(
        CHECKPOINT=CHECKPOINT,
        FRAME_GAP=FRAME_GAP,
        INFER_SIZE=INFER_SIZE,
        INFER_SCALE=INFER_SCALE,
      )
      json.dump(meta, f)

    path = './nersemble/018/sequences/EMO-1-shout+laugh/images/cam_220700191.mp4'

    # Get number of frames
    frames_no = None
    with iio.imopen(path, 'r', plugin='pyav') as f:
      prop = f.properties()
      frames_no = prop.n_images

    # Load video
    frame_iter = iio.imiter(path, plugin='pyav')
    # frame_iter = itt.islice(frame_iter, 2)
    frame_iter = enumerate(frame_iter)
    frame_iter = tqdm(frame_iter, total=frames_no)

    # Set up frame deque
    maxlen = FRAME_GAP + 2
    dq = Deque(maxlen=maxlen)

    pool: ThreadPool = stack.enter_context(ThreadPool(max_workers=32))

    with open(CSV_PATH, 'w') as f:
      writer = csv.DictWriter(f, ['video', 'frames', 'loss_mae', 'loss_mse', 'delta_time'])
      writer.writeheader()
      writer_lock = threading.Lock()

      for frame in frame_iter:
        dq.append(frame)
        if len(dq) < maxlen:
          continue

        fr0, img0 = dq[ 0]
        fr1, img1 = dq[-1]

        pool.submit(thread_frame, fr0, img0, fr1, img1)

if __name__ == '__main__':
  main()
