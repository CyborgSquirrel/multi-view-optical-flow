from pathlib import Path

import plotnine as pn
import polars as pl
from tqdm import tqdm

from ml_util import mag_image_to_array
from unimatch.unimatch import flow_warp
from util import pipe


def main():
  df = pl.read_ipc("warp_err.arrow")
  print(df)

  # df_agg = df.group_by("output", "frame").agg(pl.col("error").mean())
  # df_agg = df.group_by("frame", "output").agg(
  #   pl.col("error").mean(),
  #   (pl.col("error").mean() - 1.96 * pl.col("error").std()).alias("error_min"),
  #   (pl.col("error").mean() + 1.96 * pl.col("error").std()).alias("error_max"),
  # )

  # (pn.ggplot(df_agg, pn.aes(x="frame", y="error", color="output"))
  #   # + pn.geom_point()
  #   + pn.geom_line()
  #   + pn.geom_ribbon(pn.aes(ymin="error_min", ymax="error_max"), alpha=0.2)
  #   # + pn.theme_minimal()
  # ).show()

  (pn.ggplot(df, pn.aes(y="error", color="output"))
    + pn.geom_boxplot()
    # + pn.theme(
    #   axis_text_x=pn.element_blank(),
    #   axis_title_x=pn.element_blank(),
    #   axis_ticks_x=pn.element_blank(),
    #   # axis_text_y=pn.element_blank(),
    #   # axis_title_y=pn.element_blank(),
    #   # axis_ticks_y=pn.element_blank(),
    # )
    + pn.theme(figure_size=(2, 6), dpi=300)
    # + pn.scale_y_log10()
    + pn.ylim(0, None)
  ).save("warp-err-box.png")

  df_agg = df.group_by("frame", "output").agg(
    pl.col("error").mean(),
    pl.col("error").min().alias("error_min"),
    pl.col("error").max().alias("error_max"),
  )
  print(df_agg)

  (pn.ggplot(df_agg, pn.aes(x="frame", color="output"))
    + pn.geom_line(pn.aes(y="error"))
    + pn.geom_line(pn.aes(y="error_min"), alpha=0.3)
    + pn.geom_line(pn.aes(y="error_max"), alpha=0.3)
    + pn.theme(figure_size=(8, 6), dpi=300)
    + pn.ylim(0, None)
  ).save("warp-err-line.png")

if __name__ == "__main__":
  main()
