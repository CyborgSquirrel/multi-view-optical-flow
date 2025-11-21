import argparse
import contextlib as ctl
import logging
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ml_util import (flow_to_image_hue, unimatch_get_transform,
                     unimatch_load_img)
from util import deps, make_setup_subparser, osp, proc, resolve_local_path

logger = logging.getLogger(__name__)

try:
  from unimatch.unimatch import UniMatch, flow_warp
except ModuleNotFoundError:
  logger.warning("Unimatch not setup, can't import it...")

# Set this because otherwise PyTorch hangs forever sometimes
os.environ['OMP_NUM_THREADS'] = '1'

IS_NIXOS = osp.exists('/nix')

# Fix linker errors using nix-ld when running on NixOS
if IS_NIXOS:
  os.environ['LD_LIBRARY_PATH'] = os.environ['NIX_LD_LIBRARY_PATH']

parser = ArgumentParser()
subparsers = parser.add_subparsers(dest='action', required=True)
setup_subparser = make_setup_subparser(subparsers)

UNIMATCH_GIT = 'https://github.com/autonomousvision/unimatch'
UNIMATCH_REPO_DIR = 'unimatch.gen'
UNIMATCH_DIR = 'unimatch'
UNIMATCH_UTILS_DIR = 'unimatch_utils'

@setup_subparser()
@deps([UNIMATCH_DIR, UNIMATCH_UTILS_DIR], [])
def tool_unimatch_setup():
  proc('git', 'clone', UNIMATCH_GIT, UNIMATCH_REPO_DIR)
  shutil.copytree(osp.join(UNIMATCH_REPO_DIR, 'unimatch'), UNIMATCH_DIR)
  shutil.copytree(osp.join(UNIMATCH_REPO_DIR, 'utils'), UNIMATCH_UTILS_DIR)

# Model zoo is here
# https://github.com/autonomousvision/unimatch/blob/master/MODEL_ZOO.md

@setup_subparser
def subparser(subparser: ArgumentParser):
  arg = subparser.add_argument

  arg('image0', type=Path)
  arg('image1', type=Path)

  arg('-o', '--output', type=str)
  arg('-of', '--output-format', type=str, choices=['image', 'raw'], default='image')
  arg('-ow', '--output-warped', action='store_true')

  arg('-c', '--checkpoint', type=str, required=True)

  arg('--padding-factor', default=16, type=int,
      help='the input should be divisible by padding_factor, otherwise do padding or resizing')
  arg('--infer-scale', default=1, type=float)

  arg('--task', default='flow', choices=['flow', 'stereo', 'depth'], type=str)
  arg('--num-scales', default=1, type=int,
      help='feature scales: 1/8 or 1/8 + 1/4')
  arg('--feature-channels', default=128, type=int)
  arg('--upsample-factor', default=8, type=int)
  arg('--num-head', default=1, type=int)
  arg('--ffn-dim-expansion', default=4, type=int)
  arg('--num-transformer-layers', default=6, type=int)
  arg('--reg-refine', action='store_true',
      help='optional task-specific local regression refinement')

  arg('--attn-type', default='swin', type=str,
      help='attention function')
  arg('--attn-splits-list', default=[2], type=int, nargs='+',
      help='number of splits in attention')
  arg('--corr-radius-list', default=[-1], type=int, nargs='+',
      help='correlation radius for matching, -1 indicates global matching')
  arg('--prop-radius-list', default=[-1], type=int, nargs='+',
      help='self-attention radius for propagation, -1 indicates global attention')
  arg('--num-reg-refine', default=1, type=int,
      help='number of additional local regression refinement')
@subparser
@deps([], [tool_unimatch_setup])
def tool_unimatch_run():
  global parser_args
  pa = parser_args

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Initialize model
  model = UniMatch(
    feature_channels=pa.feature_channels,
    num_scales=pa.num_scales,
    upsample_factor=pa.upsample_factor,
    num_head=pa.num_head,
    ffn_dim_expansion=pa.ffn_dim_expansion,
    num_transformer_layers=pa.num_transformer_layers,
    reg_refine=pa.reg_refine,
    task=pa.task,
  ).to(device)
  model.eval()

  # Load checkpoint
  checkpoint = resolve_local_path(pa.checkpoint)
  checkpoint = torch.load(checkpoint, map_location=device)
  model.load_state_dict(checkpoint['model'])

  img0 = unimatch_load_img(pa.image0)
  img1 = unimatch_load_img(pa.image1)

  trans_fn, restore_fn = unimatch_get_transform(
    [img0, img1],
    infer_scale=pa.infer_scale,
    padding_factor=pa.padding_factor,
  )

  with torch.no_grad():
    results_dict = model(
      # Backward flow
      trans_fn(img1).unsqueeze(0),
      trans_fn(img0).unsqueeze(0),

      # Forward flow
      # trans_fn(img0).unsqueeze(0),
      # trans_fn(img1).unsqueeze(0),

      attn_type=pa.attn_type,
      attn_splits_list=pa.attn_splits_list,
      corr_radius_list=pa.corr_radius_list,
      prop_radius_list=pa.prop_radius_list,
      num_reg_refine=pa.num_reg_refine,
      task='flow',
    )

  flow = results_dict['flow_preds'][-1]  # [B, 2, H, W]
  flow = flow.squeeze(0)
  flow = restore_fn(flow, scale_flow=True)  # [2, H, W]

  if pa.output_warped:
    # img1w = forward_warp(
    #   img0.unsqueeze(0),
    #   flow.permute(1, 2, 0).unsqueeze(0),
    #   1,
    # )
    img1w = flow_warp(img0.unsqueeze(0), flow.unsqueeze(0)).squeeze(0)

    img1w = img1w.squeeze(0)
    img1w = img1w.permute(1, 2, 0)
    img1w = img1w.cpu().numpy()
    img1w = img1w.astype(np.uint8)
    img1w = Image.fromarray(img1w)
    img1w.show()

  flow = flow.permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

  # Present output
  match pa.output_format:
    case 'raw':
      if pa.output is None:
        raise RuntimeError(f'Must pass output file for {parser_args.output_format=}')
      torch.save(flow, pa.output)
    case 'image':
      flow = flow_to_image_hue(flow)
      if pa.output is None:
        flow.show()
      else:
        flow.save(pa.output)

@setup_subparser
def subparser(subparser: ArgumentParser):  # pylint: disable=E0102
  arg = subparser.add_argument
  arg('--radius', type=int, default=500)
@subparser
def tool_flow_wheel():
  space = np.linspace(-1, 1, num=parser_args.radius)
  x, y = np.meshgrid(space, space)

  flow = np.stack([x, y], axis=-1)
  mag = np.linalg.vector_norm(flow, axis=-1)
  flow[mag > 1, :] = 0
  flow_to_image_hue(flow).show()

NERSEMBLE_DATA_GIT = 'https://github.com/tobias-kirschstein/nersemble-data'
NERSEMBLE_DATA_REPO_DIR = 'nersemble-data.gen'
NERSEMBLE_DATA_PATH = osp.join(NERSEMBLE_DATA_REPO_DIR, '.venv', 'bin', 'nersemble-data')

@setup_subparser()
@deps([NERSEMBLE_DATA_PATH], [])
def tool_nersemble_data_setup():
  proc('git', 'clone', NERSEMBLE_DATA_GIT, NERSEMBLE_DATA_REPO_DIR)
  with ctl.chdir(NERSEMBLE_DATA_REPO_DIR):
    proc('python3', '-m', 'venv', '.venv')
    proc(osp.join('.venv', 'bin', 'pip'), 'install', '.')

@setup_subparser
def subparser(subparser: ArgumentParser):
  # NOTE(andrei): This is not ideal but whatever.
  subparser.add_argument('args', nargs=argparse.REMAINDER)
@subparser
@deps([], [tool_nersemble_data_setup])
def tool_nersemble_data():
  proc(NERSEMBLE_DATA_PATH, *parser_args.args, check=False)

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  parser_args = parser.parse_args()
  parser_args.fn()
