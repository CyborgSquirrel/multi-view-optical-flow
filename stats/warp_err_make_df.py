from pathlib import Path

import polars as pl
from tqdm import tqdm

from ml_util import mag_image_to_array
from unimatch.unimatch import flow_warp
from util import pipe

OUT_PATH_TXT = "./output/{output}-{pose}-{frame}.txt"

def main():
  pose_ids = [
    "222200044",
    "222200038",
    "222200045",
    "222200041",
    # "222200046",
    "222200039",
    "221501007",
    "222200037",
    "222200040",
    "220700191",
    "222200036",
    "222200043",
  ]

  rows = []
  # for fr0 in tqdm(range(1, 360)):
  for fr0 in tqdm(range(1, 412)):
  # for fr0 in tqdm(range(1, 140)):
    for pose_id in pose_ids:
      rows.append(dict(
        frame=fr0,
        pose=pose_id,
        output="base",
        error=float(Path(OUT_PATH_TXT.format(output="err-base", pose=pose_id, frame=fr0)).read_text()),
      ))
      rows.append(dict(
        frame=fr0,
        pose=pose_id,
        output="nersemble",
        error=float(Path(OUT_PATH_TXT.format(output="err-ners", pose=pose_id, frame=fr0)).read_text()),
      ))
      rows.append(dict(
        frame=fr0,
        pose=pose_id,
        output="unimatch",
        error=float(Path(OUT_PATH_TXT.format(output="err-unim", pose=pose_id, frame=fr0)).read_text()),
      ))

  df = pl.DataFrame({
    key: [row[key] for row in rows]
    for key in ["frame", "pose", "output", "error"]
  })
  df.write_ipc("warp_err.arrow")

if __name__ == "__main__":
  main()
