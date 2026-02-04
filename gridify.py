#!/usr/bin/env python3

import argparse
import dataclasses as dc
import itertools as itt
import json
import os.path as osp
import subprocess

from PIL import Image


class Inp:
  pass

@dc.dataclass
class FileInp(Inp):
  path: str

@dc.dataclass
class GenInp(Inp):
  color: str


def create_grid(
  inputs,
  grid_size,
  output_path,
  *,
  background_color: str,
):
  new_inputs = []
  for inp in inputs:
    if inp.startswith(":"):
      color = inp.removeprefix(":")
      if color == "":
        color = background_color
      new_inputs.append(GenInp(color))
    else:
      new_inputs.append(FileInp(inp))
  inputs = new_inputs
  del new_inputs

  cols, rows = grid_size

  # deal with formats
  def get_format(path):
    _, ext = osp.splitext(path)
    match ext.lower():
      case ".png" | ".jpg" | ".jpeg":
        return "image"
      case ".mp4":
        return "video"
      case _:
        raise RuntimeError(f"Unexpected extension {repr(ext)}")

  def formats_gen_fn():
    for inp in inputs:
      if not isinstance(inp, FileInp):
        continue
      yield get_format(inp.path)

  formats_gen = formats_gen_fn()
  format = next(formats_gen)
  if not all(curr_format == format for curr_format in formats_gen):
    raise RuntimeError("All inputs must be in the same format (i.e. 'image' or 'video')")

  if get_format(output_path) != format:
    raise RuntimeError("Output format doesn't match input")

  # sanity check
  if len(inputs) > cols * rows:
    print(f"Warning: Got {len(inputs)} images but grid is {cols}x{rows} = {cols*rows} slots")
    print(f"Only using the first {cols*rows} images")
    inputs = inputs[:cols * rows]

  for _ in range(len(inputs), cols*rows):
    inputs.append(GenInp(background_color))

  match format:
    case "image":
      # calculate cell size
      probe_w = []
      probe_h = []
      for inp in inputs:
        if not isinstance(inp, FileInp):
          continue
        img = Image.open(inp.path)
        probe_w.append(img.size[0])
        probe_h.append(img.size[1])

      w = min(probe_w)
      h = min(probe_h)

      # create images
      imgs = []
      for inp in inputs:
        match inp:
          case FileInp(path):
            imgs.append(Image.open(path))
          case GenInp(color):
            imgs.append(Image.new("RGB", (w, h), color=color))
          case _:
            raise RuntimeError(f"Unexpected {inp=}")

      # create blank canvas
      grid = Image.new("RGB", (cols * w, rows * h))
  
      # paste images into grid
      for idx, img in enumerate(imgs):
        row = idx // cols
        col = idx % cols
        x = col * w
        y = row * h
        grid.paste(img, (x, y))
  
      # Save the result
      grid.save(output_path)
      print(f"Grid created: {output_path} ({cols}x{rows}, {len(inputs)} images)")

    case "video":
      # calculate cell size + duration
      probe_w = []
      probe_h = []
      probe_d = []
      for inp in inputs:
        if not isinstance(inp, FileInp):
          continue

        path = inp.path

        if not osp.exists(path):
          raise RuntimeError(f"Nonexistent file: {repr(path)}")

        p = subprocess.run([
          "ffprobe",
          "-v", "error",
          "-select_streams", "v:0",
          "-show_entries", "stream=width,height:format=duration",
          "-of", "json",
          path,
        ], stdout=subprocess.PIPE)
        r = json.loads(p.stdout)

        w = r["streams"][0]["width"]
        h = r["streams"][0]["height"]
        d = r["format"]["duration"]

        probe_w.append(w)
        probe_h.append(h)
        probe_d.append(d)

      w = min(probe_w)
      h = min(probe_h)
      d = min(probe_d)

      # generate ffmpeg command
      cmd = ["ffmpeg"]

      # inputs
      inp_to_idx = dict()
      inp_idx_it = itt.count()
      for inp in inputs:
        if not isinstance(inp, FileInp):
          continue
        if inp.path in inp_to_idx:
          continue
        inp_to_idx[inp.path] = next(inp_idx_it)
        cmd += ["-i", inp.path]

      # filter complex
      filter_complex = []
      for i, inp in enumerate(inputs):
        match inp:
          case FileInp(path):
            idx = inp_to_idx[path]
            filter_complex.append(f"[{idx}:v]scale={w}:{h}[v{i}]")
          case GenInp(color):
            filter_complex.append(f"color=c={color}:s={w}x{h}:d={d}[v{i}]")
          case _:
            raise RuntimeError(f"Unexpeted {inp=}")

      i_iter = iter(range(rows * cols))
      for ri in range(rows):
        hstack = []
        for _ci in range(cols):
          i = next(i_iter)
          hstack.append(f"[v{i}]")
        hstack.append(f"hstack=inputs={cols}[row{ri}]")
        hstack = "".join(hstack)
        filter_complex.append(hstack)

      if rows == 1:
        out = "row0"
      else:
        out = "out"
        vstack = []
        for ri in range(rows):
          vstack.append(f"[row{ri}]")
        vstack.append(f"vstack=inputs={rows}[out]")
        vstack = "".join(vstack)
        filter_complex.append(vstack)
      filter_complex = ";".join(filter_complex)

      cmd += ["-filter_complex", filter_complex]

      # output
      cmd += ["-map", f"[{out}]", output_path]

      subprocess.run(cmd, check=True)

def main():
  parser = argparse.ArgumentParser(description="Create an image grid from multiple images")
  parser.add_argument("--size", required=True, help="Grid size as COLSxROWS (e.g., 4x3)")
  parser.add_argument("--output", "-o", help="Output file path")
  parser.add_argument("-bg", "--background-color", default="black")
  parser.add_argument("inputs", nargs="+", help="Input image files")
  
  args = parser.parse_args()
  
  # parse grid size
  try:
    cols, rows = args.size.split("x")
    cols, rows = int(cols), int(rows)
  except ValueError:
    parser.error("Grid size must be in format COLSxROWS (e.g., 4x3)")
  
  create_grid(args.inputs, (cols, rows), args.output, background_color=args.background_color)

if __name__ == '__main__':
  main()
