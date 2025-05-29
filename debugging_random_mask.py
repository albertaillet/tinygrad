#!/usr/bin/env python3

# tinygrad implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/
# https://siboehm.com/articles/22/CUDA-MMM
import random
from tinygrad import dtypes, Tensor
from tinygrad.helpers import getenv
import numpy as np
from PIL import Image
from term_image.image import AutoImage
from einops import rearrange

def set_seed(seed):
  Tensor.manual_seed(seed)
  random.seed(seed)

def make_square_mask(shape, mask_size) -> Tensor:
  BS, _, H, W = shape
  low_x = Tensor.randint(BS, low=0, high=W-mask_size).reshape(BS,1,1,1)
  low_y = Tensor.randint(BS, low=0, high=H-mask_size).reshape(BS,1,1,1)
  idx_x = Tensor.arange(W, dtype=dtypes.int32).reshape((1,1,1,W))
  idx_y = Tensor.arange(H, dtype=dtypes.int32).reshape((1,1,H,1))
  return (idx_x >= low_x) * (idx_x < (low_x + mask_size)) * (idx_y >= low_y) * (idx_y < (low_y + mask_size))

def to_grid(x, ih: int, iw: int):
  '''Rearranges a array of images with shape (n, c, h, w) to a grid of shape (c, ih*h, iw*w)'''
  return rearrange(x, '(ih iw) c h w -> (ih h) (iw w) c', ih=ih, iw=iw)

if __name__ == "__main__":
  set_seed(209)
  num_images, mask_size = getenv("BS", 10), getenv("MASK_SIZE", 32)
  shape = (num_images, 3, 32, 32)  # BS, C, H, W
  # mask = make_square_mask(shape, mask_size).expand((-1,3,-1,-1))
  # images_per_row = 5
  # print(mask.shape)
  # full_image = to_grid(mask.numpy(), num_images, 1)  # Convert to grid of images
  # print(full_image.shape)
  # img = Image.fromarray((full_image * 255).astype(np.uint8))
  # AutoImage(img).draw()
  for _ in range(10):
    mask = make_square_mask(shape, mask_size).numpy()
    print(mask.mean())
    assert mask.all(), "Mask should be all True or all False"