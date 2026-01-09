import argparse
import contextlib as ctl
import functools as ft
import itertools as itt
import logging
import os
import random
import shutil
import zipfile
from argparse import ArgumentParser
from collections import ChainMap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar, Optional, Self
from zipfile import ZipFile

import humanize
import imageio as iio
import json5
import numpy as np
import torch
import torch.nn.functional as F
from colab_util import IN_COLAB, PROJECT_PATH
from ml_util import (flow_to_image_hue, forward_warp, image_to_flow_hue,
                     load_img, unimatch_get_transform, unimatch_load_img)
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm
from util import deps, make_setup_subparser, osp, proc, resolve_local_path

logger = logging.getLogger(__name__)

try:
  from unimatch.unimatch import UniMatch, flow_warp
except ModuleNotFoundError:
  logger.warning("Unimatch not setup, can't import it...")

# For most everything we need to run in REPO_ROOT
REPO_ROOT = osp.dirname(__file__)
run_dir = os.getcwd()
os.chdir(REPO_ROOT)

# Set this because otherwise PyTorch hangs forever sometimes
os.environ["OMP_NUM_THREADS"] = "1"

IS_NIXOS = osp.exists("/nix")

# Fix linker errors using nix-ld when running on NixOS
if IS_NIXOS:
  os.environ["LD_LIBRARY_PATH"] = os.environ["NIX_LD_LIBRARY_PATH"]

############################################################
#                          Config                          #
############################################################

class BaseConfig(BaseModel):
  PATH: ClassVar[str]

  @classmethod
  def get(cls) -> Self:
    if not osp.exists(cls.PATH):
      logger.warning("Config does not exist, generating default at %r...", cls.PATH)
      obj = cls()
      with open(cls.PATH, "w") as f:
        json5.dump(obj.model_dump(mode="json"), f, indent=2)

    with open(cls.PATH) as f:
      obj = json5.load(f)
    obj = cls.model_validate(obj)
    return obj

class Config(BaseConfig):
  PATH = osp.join(REPO_ROOT, "config.json5")

  DEPLOY_PATH: Optional[Path] = None

class Secrets(BaseConfig):
  PATH = osp.join(REPO_ROOT, "config.secret.json5")

  NERSEMBLE_DATA_URL: Optional[str] = None

cfg = Config.get()
sec = Secrets.get()

############################################################
#                     Commandline Args                     #
############################################################

parser = ArgumentParser()
subparsers = parser.add_subparsers(dest="action", required=True)
setup_subparser = make_setup_subparser(subparsers)

############################################################
#                          Colab                           #
############################################################

@ft.cache
def get_file_paths():
  out = proc(
    "fd", "--hidden", "--type=file",
    stdout=True,
  )
  file_paths = out.split()
  return file_paths

@setup_subparser()
def tool_deploy_list():
  file_paths = get_file_paths()
  print("\n".join(file_paths))

def write_ar_code(dst: str | Path):
  logger.info("Generating code archive...")

  if isinstance(dst, str):
    dst = Path(dst)

  dir_name = dst.name
  if (dot := dst.name.find(".")) != -1:
    dir_name = dir_name[:dot]

  file_paths = get_file_paths()
  with ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as z:
    code_id = random.randbytes(4).hex()
    logger.info("Setting code archive ID to %r...", code_id)
    z.writestr(osp.join(dir_name, "id.txt"), code_id)

    for file_path in tqdm(file_paths, desc="Zipping files..."):
      z.write(
        file_path,
        osp.join(dir_name, file_path),
      )

  stat = dst.stat()
  logger.info("Generated code archive sized %s", humanize.naturalsize(stat.st_size))

@setup_subparser()
def tool_deploy_code():
  logger.info("Making sure project directory exists...")
  cfg.DEPLOY_PATH.mkdir(parents=True, exist_ok=True)

  with TemporaryDirectory() as tmp_dir:
    tmp_dir = Path(tmp_dir)

    ar_path = tmp_dir / "code.zip"
    write_ar_code(ar_path)

    logger.info("Uploading code archive...")
    shutil.move(ar_path, cfg.DEPLOY_PATH / ar_path.name)

@setup_subparser()
def tool_deploy_nb():
  cfg.DEPLOY_PATH.mkdir(parents=True, exist_ok=True)
  shutil.copy("main.ipynb", cfg.DEPLOY_PATH)

@setup_subparser()
def tool_deploy_secrets():
  cfg.DEPLOY_PATH.mkdir(parents=True, exist_ok=True)
  shutil.copy(sec.PATH, cfg.DEPLOY_PATH)

############################################################
#                         Unimatch                         #
############################################################

UNIMATCH_GIT = "https://github.com/autonomousvision/unimatch"
UNIMATCH_REPO_DIR = "unimatch.gen"
UNIMATCH_DIR = "unimatch"
UNIMATCH_UTILS_DIR = "unimatch_utils"

@setup_subparser()
@deps([UNIMATCH_DIR, UNIMATCH_UTILS_DIR], [])
def tool_unimatch_setup():
  proc("git", "clone", UNIMATCH_GIT, UNIMATCH_REPO_DIR)
  shutil.copytree(osp.join(UNIMATCH_REPO_DIR, "unimatch"), UNIMATCH_DIR)
  shutil.copytree(osp.join(UNIMATCH_REPO_DIR, "utils"), UNIMATCH_UTILS_DIR)

# Model zoo is here
# https://github.com/autonomousvision/unimatch/blob/master/MODEL_ZOO.md

@setup_subparser
def subparser(subparser: ArgumentParser):
  arg = subparser.add_argument

  arg("image0", type=Path)
  arg("image1", type=Path)

  arg("-o", "--output", type=str)
  arg("-of", "--output-format", type=str, choices=["image", "raw"], default="image")
  arg("-ow", "--output-warped", action="store_true")

  arg("-c", "--checkpoint", type=str, required=True)

  arg("--image-scale", default=1, type=float)

  arg("--padding-factor", default=16, type=int,
      help="the input should be divisible by padding_factor, otherwise do padding or resizing")
  arg("--flow-scale", default=1, type=float)

  arg("--task", default="flow", choices=["flow", "stereo", "depth"], type=str)
  arg("--num-scales", default=1, type=int,
      help="feature scales: 1/8 or 1/8 + 1/4")
  arg("--feature-channels", default=128, type=int)
  arg("--upsample-factor", default=8, type=int)
  arg("--num-head", default=1, type=int)
  arg("--ffn-dim-expansion", default=4, type=int)
  arg("--num-transformer-layers", default=6, type=int)
  arg("--reg-refine", action="store_true",
      help="optional task-specific local regression refinement")

  arg("--attn-type", default="swin", type=str,
      help="attention function")
  arg("--attn-splits-list", default=[2], type=int, nargs="+",
      help="number of splits in attention")
  arg("--corr-radius-list", default=[-1], type=int, nargs="+",
      help="correlation radius for matching, -1 indicates global matching")
  arg("--prop-radius-list", default=[-1], type=int, nargs="+",
      help="self-attention radius for propagation, -1 indicates global attention")
  arg("--num-reg-refine", default=1, type=int,
      help="number of additional local regression refinement")
@subparser
@deps([], [tool_unimatch_setup])
def tool_unimatch_run():
  global parser_args
  pa = parser_args

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
  model.load_state_dict(checkpoint["model"])

  img0 = load_img(pa.image0, output_type=np.ndarray)
  img1 = load_img(pa.image1, output_type=np.ndarray)

  img0 = unimatch_load_img(img0).to(device)
  img1 = unimatch_load_img(img1).to(device)

  if pa.image_scale != 1:
    img0 = F.interpolate(
      img0.unsqueeze(0),
      size=[round(a*pa.image_scale) for a in img0.shape[-2:]],
      mode="bilinear",
      align_corners=True,
    ).squeeze(0)
    img1 = F.interpolate(
      img1.unsqueeze(0),
      size=[round(a*pa.image_scale) for a in img1.shape[-2:]],
      mode="bilinear",
      align_corners=True,
    ).squeeze(0)

  trans_fn, restore_fn = unimatch_get_transform(
    [img0, img1],
    flow_scale=pa.flow_scale,
    padding_factor=pa.padding_factor,
  )

  Image.fromarray(img0.permute(1, 2, 0).to(torch.uint8).cpu().numpy()).show()
  Image.fromarray(img1.permute(1, 2, 0).to(torch.uint8).cpu().numpy()).show()

  logger.info("Running model...")
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
      task="flow",
    )

  flow = results_dict["flow_preds"][-1]  # [B, 2, H, W]
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
    case "raw":
      if pa.output is None:
        raise RuntimeError(f"Must pass output file for {parser_args.output_format=}")
      torch.save(flow, pa.output)
    case "image":
      flow = flow_to_image_hue(flow)
      if pa.output is None:
        flow.show()
      else:
        flow.save(pa.output, exif=flow.getexif())

############################################################
#                      Nersemble Data                      #
############################################################

NERSEMBLE_DATA_GIT = "https://github.com/tobias-kirschstein/nersemble-data"
NERSEMBLE_DATA_REPO_DIR = "nersemble-data-tool.gen"
NERSEMBLE_DATA_PATH = osp.join(NERSEMBLE_DATA_REPO_DIR, ".venv", "bin", "nersemble-data")

NERSEMBLE_DATA_ENV_PATH = Path.home() / ".config" / "nersemble_data" / ".env"

if not IN_COLAB:
  deps_out = [NERSEMBLE_DATA_PATH]
else:
  deps_out = []

@setup_subparser()
@deps(deps_out, [])
def tool_nersemble_data_setup():
  with deps.outer, deps([NERSEMBLE_DATA_REPO_DIR], []):
    proc("git", "clone", NERSEMBLE_DATA_GIT, NERSEMBLE_DATA_REPO_DIR)

  if sec.NERSEMBLE_DATA_URL is not None:
    NERSEMBLE_DATA_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NERSEMBLE_DATA_ENV_PATH.open("w") as f:
      f.write(f"NERSEMBLE_DATA_URL=\"{sec.NERSEMBLE_DATA_URL}\"\n")

  with ctl.chdir(NERSEMBLE_DATA_REPO_DIR):
    if not IN_COLAB:
      proc("python3", "-m", "venv", ".venv")
      proc(osp.join(".venv", "bin", "pip"), "install", ".")
    else:
      # NOTE(andrei): Venv doesn't seem to work in Colab, so just installing it
      # raw.
      proc("pip", "install", ".")

@setup_subparser
def subparser(subparser: ArgumentParser):
  # NOTE(andrei): This is not ideal but whatever.
  subparser.add_argument("args", nargs=argparse.REMAINDER)
@subparser
@deps([], [tool_nersemble_data_setup])
def tool_nersemble_data():
  if not IN_COLAB:
    proc(NERSEMBLE_DATA_PATH, *parser_args.args, check=False)
  else:
    proc("nersemble-data", *parser_args.args, check=False)

############################################################
#                        Nersemble                         #
############################################################

# !curl --progress-bar 'https://nextcloud.tobias-kirschstein.de/index.php/s/gQoLTHjQkNNHN2j/download?path=%2FNERS-9018' > 'NERS-9018-x.zip'

# NOTE(andrei): Using my fork to escape Conda.
# NERSEMBLE_GIT = "https://github.com/tobias-kirschstein/nersemble"
NERSEMBLE_GIT = "https://github.com/CyborgSquirrel/nersemble"
NERSEMBLE_REPO_DIR = "nersemble.gen"

@setup_subparser()
@deps([], [])
def tool_nersemble_setup():
  with deps.outer, deps([NERSEMBLE_REPO_DIR], [], name="nersemble_clone"):
    # NOTE(andrei): Using --depth 1 because otherwise it takes ages.
    proc("git", "clone", "--depth", "1", NERSEMBLE_GIT, NERSEMBLE_REPO_DIR)

  if IN_COLAB:
    # Grab pre-built tinycudann
    name = "tinycudann-2.0-cp312-cp312-linux_x86_64.whl"
    src = osp.join(PROJECT_PATH, name)
    dst = osp.join(NERSEMBLE_REPO_DIR, name)
    with deps.outer, deps([dst], [src]):
      shutil.copy(src, dst)

  with ctl.chdir(NERSEMBLE_REPO_DIR):
    proc("./install.sh")

############################################################
#                           Misc                           #
############################################################

@setup_subparser
def subparser(subparser: ArgumentParser):
  arg = subparser.add_argument
  arg("image", type=Path)
@subparser
def tool_img_show():
  image = load_img(parser_args.image)
  Image.fromarray(image.to(torch.uint8).numpy()).show()

@setup_subparser
def subparser(subparser: ArgumentParser):
  arg = subparser.add_argument
  arg("vid0", type=Path)
  arg("vid1", type=Path)
  arg("output", type=Path)
@subparser
def tool_vid_diff():
  from skimage.transform import resize
  from tqdm import tqdm

  vid0_iter = iio.imiter(parser_args.vid0, plugin="pyav")
  vid1_iter = iio.imiter(parser_args.vid1, plugin="pyav")

  with ctl.closing(iio.get_writer(parser_args.output, fps=73)) as output_writer:
    for fr0, fr1 in tqdm(zip(vid0_iter, vid1_iter)):
      h = min(fr0.shape[0], fr1.shape[0])
      w = min(fr0.shape[1], fr1.shape[1])

      fr0 = resize(fr0, (h, w))
      fr1 = resize(fr1, (h, w))

      err = np.abs(fr0 - fr1)
      err = np.mean(err, axis=-1)
      err = np.clip(err, 0, 255)
      output_writer.append_data(err)

############################################################
#                           Flow                           #
############################################################

@setup_subparser
def subparser(subparser: ArgumentParser):
  arg = subparser.add_argument
  arg("image", type=Path)
  arg("flow", type=Path)
  arg("--flow-mul", type=float, default=1)
  arg("-o", "--output", type=str)
@subparser
def tool_flow_warp():
  flow = image_to_flow_hue(parser_args.flow)
  flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
  flow = flow * parser_args.flow_mul

  image = load_img(parser_args.image)
  image = image.permute(2, 0, 1).unsqueeze(0)
  image = F.interpolate(
    image,
    size=flow.shape[-2:],
    mode="bilinear",
    align_corners=True,
  )

  if parser_args.output is None:
    Image.fromarray(image.squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()).show()

  image_warped = flow_warp(
    image.float(),
    flow.float(),
  )

  image_warped = image_warped.round().clip(0, 255).to(torch.uint8)
  image_warped = Image.fromarray(image_warped.squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy())

  if parser_args.output is None:
    image_warped.show()
  else:
    image_warped.save(parser_args.output)

@setup_subparser
def subparser(subparser: ArgumentParser):  # pylint: disable=E0102
  arg = subparser.add_argument
  arg("--radius", type=int, default=500)
@subparser
def tool_flow_wheel():
  space = np.linspace(-1, 1, num=parser_args.radius)
  x, y = np.meshgrid(space, space)

  flow = np.stack([x, y], axis=-1)
  mag = np.linalg.vector_norm(flow, axis=-1)
  flow[mag > 1, :] = 0
  flow_to_image_hue(flow).show()

############################################################
#                           Main                           #
############################################################

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  parser_args = parser.parse_args()
  parser_args.fn()
