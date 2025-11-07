import argparse
import logging
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import requests
import torch

from util import ctx_chdir, deps, make_setup_subparser, osp, proc

logger = logging.getLogger(__name__)

try:
  from unimatch.unimatch import UniMatch
except ModuleNotFoundError:
  logger.warning("Unimatch not setup, can't import it...")

IS_NIXOS = osp.exists('/nix')

# Fix linker errors using nix-ld when running on NixOS
if IS_NIXOS:
  os.environ['LD_LIBRARY_PATH'] = os.environ['NIX_LD_LIBRARY_PATH']

parser = ArgumentParser()
subparsers = parser.add_subparsers(dest='action', required=True)
setup_subparser = make_setup_subparser(subparsers)

MODEL_PATH = 'unimatch.pth'
MODEL_URL = 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-mixdata-train320x576-9ff1c094.pth'

@setup_subparser()
@deps([MODEL_PATH], [])
def tool_unimatch_dl_model():
  req = requests.get(MODEL_URL)
  with open(MODEL_PATH, 'wb') as f:
    f.write(req.content)

UNIMATCH_GIT = 'https://github.com/autonomousvision/unimatch'
UNIMATCH_DIR = 'unimatch.gen'

@setup_subparser()
@deps([], [])
def tool_unimatch_setup():
  proc('git', 'clone', UNIMATCH_GIT, UNIMATCH_DIR)
  shutil.copytree(osp.join(UNIMATCH_DIR, 'unimatch'), 'unimatch')

@setup_subparser
def subparser(subparser: ArgumentParser):
  arg = subparser.add_argument

  arg('image1', type=Path)
  arg('image2', type=Path)

  arg('--task', default='flow', choices=['flow', 'stereo', 'depth'], type=str)
  arg('--num_scales', default=1, type=int,
      help='feature scales: 1/8 or 1/8 + 1/4')
  arg('--feature_channels', default=128, type=int)
  arg('--upsample_factor', default=8, type=int)
  arg('--num_head', default=1, type=int)
  arg('--ffn_dim_expansion', default=4, type=int)
  arg('--num_transformer_layers', default=6, type=int)
  arg('--reg_refine', action='store_true',
      help='optional task-specific local regression refinement')
@subparser
def tool_unimatch_run():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = UniMatch(
    feature_channels=parser_args.feature_channels,
    num_scales=parser_args.num_scales,
    upsample_factor=parser_args.upsample_factor,
    num_head=parser_args.num_head,
    ffn_dim_expansion=parser_args.ffn_dim_expansion,
    num_transformer_layers=parser_args.num_transformer_layers,
    reg_refine=parser_args.reg_refine,
    task=parser_args.task,
  ).to(device)

  results_dict = model(
    image1, image2,
    # attn_type=attn_type,
    # attn_splits_list=attn_splits_list,
    # corr_radius_list=corr_radius_list,
    # prop_radius_list=prop_radius_list,
    # num_reg_refine=num_reg_refine,
    task='flow',
  )

NERSEMBLE_DATA_GIT = 'https://github.com/tobias-kirschstein/nersemble-data'
NERSEMBLE_DATA_DIR = 'nersemble-data.gen'
NERSEMBLE_DATA_PATH = osp.join(NERSEMBLE_DATA_DIR, '.venv', 'bin', 'nersemble-data')

@setup_subparser()
@deps([NERSEMBLE_DATA_PATH], [])
def tool_nersemble_data_setup():
  proc('git', 'clone', NERSEMBLE_DATA_GIT, NERSEMBLE_DATA_DIR)
  with ctx_chdir(NERSEMBLE_DATA_DIR):
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
