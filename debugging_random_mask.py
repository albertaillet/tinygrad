#!/usr/bin/env python3
from os import getenv
import numpy as np
from PIL import Image

def make_mask(shape, mask_size:int) -> np.ndarray:
  BS, _, H, W = shape
  low_x = np.random.randint(size=BS, low=0, high=W-mask_size).reshape(BS,1,1,1)
  low_y = np.random.randint(size=BS, low=0, high=H-mask_size).reshape(BS,1,1,1)
  idx_x = np.arange(W).reshape((1,1,1,W))
  idx_y = np.arange(H).reshape((1,1,H,1))
  return (idx_x >= low_x) * (idx_x < (low_x + mask_size)) * (idx_y >= low_y) * (idx_y < (low_y + mask_size))

def to_grid(x:np.ndarray, ih: int, iw: int) -> np.ndarray:
    BS, C, H, W = x.shape  # x is of shape (ih*iw, c, h, w)
    assert BS == ih * iw, "Batch size must equal ih * iw"
    return x.reshape(ih, iw, C, H, W).transpose(0, 3, 1, 4, 2).reshape(ih * H, iw * W, C)

if __name__ == "__main__":
  BS, mask_size, images_per_row = int(getenv("BS", 300)), int(getenv("MASK_SIZE", 32)), int(getenv("ROW", 25))
  shape = (BS, 3, 36, 36)
  mask = make_mask(shape, mask_size)
  full_image = to_grid(mask, BS // images_per_row, images_per_row).repeat(repeats=3, axis=-1)
  Image.fromarray((full_image * 255).astype(np.uint8)).show()