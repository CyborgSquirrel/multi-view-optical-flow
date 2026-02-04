import matplotlib.pyplot as plt
import plotnine as pn
import polars as pl

df = pl.read_ipc("flow_err.arrow")
print(df)

# (pn.ggplot(df.filter(pl.col("interp").eq("lin")), pn.aes(y="mae"))
#   + pn.geom_boxplot()
# ).show()

(pn.ggplot(df, pn.aes(y="mae"))
  + pn.geom_boxplot()
  + pn.facet_wrap("interp", ncol=1)
  + pn.theme(
    axis_text_x=pn.element_blank(),
    axis_title_x=pn.element_blank(),
    axis_ticks_x=pn.element_blank(),
    # axis_text_y=pn.element_blank(),
    # axis_title_y=pn.element_blank(),
    # axis_ticks_y=pn.element_blank(),
  )

  + pn.ylim(0, 1)
  + pn.theme(figure_size=(1, 6), dpi=300)
).save("flow-err-box.png")

# df2 = df.filter(pl.col("interp").eq("lin")).filter(pl.col("pose").eq("220700191"))

df_agg = df.group_by("interp", "frame").agg(pl.col("mae").mean())

(pn.ggplot(df, pn.aes(x="frame", y="mae"))
  + pn.geom_point(size=0.01, color="gray")

  + pn.geom_smooth(data=df_agg, span=0.1)
  # + pn.geom_smooth(method="lm")
  # + pn.geom_smooth(data=df.group_by(""))

  + pn.facet_wrap("interp", ncol=1)

  + pn.theme(figure_size=(6, 6), dpi=300)
  + pn.ylim(0, 1)

  + pn.theme(
    axis_text_x=pn.element_blank(),
    axis_title_x=pn.element_blank(),
    axis_ticks_x=pn.element_blank(),
    # axis_text_y=pn.element_blank(),
    # axis_title_y=pn.element_blank(),
    # axis_ticks_y=pn.element_blank(),
  )
).save("flow-err-line.png")

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))

# p1.draw(ax1)
# p2.draw(ax2)
# plt.tight_layout()
# plt.show()
