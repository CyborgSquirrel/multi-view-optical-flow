import polars as pl
import numpy as np
from scipy.interpolate import griddata
from skimage.transform import resize  # pylint: disable
from tqdm import tqdm

from ml_util import mag_image_to_array
from unimatch.unimatch import flow_warp
from util import pipe

ERR_PATH = "./flow-err/{output}-{pose}-{frame}.png"

def main():
  pose_ids = [
    # "221501007",

    "220700191",
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

  dst_pose_id = "221501007"

  rows = []
  for fr0 in tqdm(range(1, 413)):
    for pose_id in pose_ids:
      lin_path = ERR_PATH.format(output="lin", pose=pose_id, frame=fr0)
      lin_err = mag_image_to_array(lin_path)
      lin_mae = np.mean(np.abs(lin_err))
      rows.append(dict(frame=fr0, pose=pose_id, interp="linear", mae=lin_mae))

      bin_path = ERR_PATH.format(output="bin", pose=pose_id, frame=fr0)
      bin_err = mag_image_to_array(bin_path)
      bin_mae = np.mean(np.abs(bin_err))
      rows.append(dict(frame=fr0, pose=pose_id, interp="binned", mae=bin_mae))

  df = pl.DataFrame({
    key: [row[key] for row in rows]
    for key in ["frame", "pose", "interp", "mae"]
  })
  df.write_ipc("flow_err.arrow")

if __name__ == "__main__":
  main()

