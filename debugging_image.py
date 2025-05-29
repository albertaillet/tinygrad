#!/usr/bin/env python3

# tinygrad implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/
# https://siboehm.com/articles/22/CUDA-MMM
import random
from tinygrad import dtypes, Tensor
from tinygrad.helpers import getenv
from tinygrad.nn import datasets
import numpy as np
from PIL import Image
from term_image.image import AutoImage

def set_seed(seed):
  Tensor.manual_seed(seed)
  random.seed(seed)

# ========== Preprocessing ==========
# return a binary mask in the format of BS x C x H x W where H x W contains a random square mask
def make_square_mask(shape, mask_size) -> Tensor:
  BS, _, H, W = shape
  low_x = Tensor.randint(BS, low=0, high=W-mask_size).reshape(BS,1,1,1)
  low_y = Tensor.randint(BS, low=0, high=H-mask_size).reshape(BS,1,1,1)
  idx_x = Tensor.arange(W, dtype=dtypes.int32).reshape((1,1,1,W))
  idx_y = Tensor.arange(H, dtype=dtypes.int32).reshape((1,1,H,1))
  return (idx_x >= low_x) * (idx_x < (low_x + mask_size)) * (idx_y >= low_y) * (idx_y < (low_y + mask_size))

def random_crop(X:Tensor, crop_size=32):
  mask = make_square_mask(X.shape, crop_size)
  
  mask = mask.expand((-1,3,-1,-1))
  X_cropped = Tensor(X.numpy()[mask.numpy()])
  return X_cropped.reshape((-1, 3, crop_size, crop_size))

def cutmix(X:Tensor, Y:Tensor, mask_size=3):
  # fill the square with randomly selected images from the same batch
  mask = make_square_mask(X.shape, mask_size)
  # order = list(range(0, X.shape[0]))
  # random.shuffle(order)
  order = Tensor.randperm(X.shape[0], device=X.device, dtype=dtypes.int32).numpy()
  X_patch = Tensor(X.numpy()[order], device=X.device, dtype=X.dtype)
  Y_patch = Tensor(Y.numpy()[order], device=Y.device, dtype=Y.dtype)
  print(Y_patch.shape)
  X_cutmix = mask.where(X_patch, X)
  mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
  Y_cutmix = mix_portion * Y_patch + (1. - mix_portion) * Y
  return X_cutmix, Y_cutmix

# ========== My modification below ==========

def random_crop_no_indexing(X:Tensor, crop_size=32):
  mask = make_square_mask(X.shape, crop_size)
  return X.masked_select(mask).reshape((-1, 3, crop_size, crop_size))

def cutmix_no_indexing(X:Tensor, Y:Tensor, mask_size=3):
  # fill the square with randomly selected images from the same batch
  mask = make_square_mask(X.shape, mask_size)
  order = Tensor.randperm(X.shape[0], device=X.device, dtype=dtypes.int32)
  X_patch, Y_patch = X[order], Y[order]
  X_cutmix = mask.where(X_patch, X)
  mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
  Y_cutmix = mix_portion * Y_patch + (1. - mix_portion) * Y
  return X_cutmix, Y_cutmix

# ====================

if __name__ == "__main__":
  mask_size = getenv("MASK_SIZE", 3) # Size of the square mask for cutmix
  seed = getenv("SEED", 209) # Default seed for reproducibility
  # Example usage of the functions
  n_images = getenv("N_IMAGES", 8) # Number of images in the batch
  image_size=getenv("IMAGE_SIZE", 32)
  # X = Tensor.randn(n_images, 3, image_size, image_size) # Simulated batch of images
  Y = Tensor.randn(n_images, 10) # Simulated labels
  X, _, _, _ = datasets.cifar()  # Load CIFAR-10 dataset
  X = X[:n_images]  # Limit to n_images
  # Y = Y[:n_images]  # Limit to n_images
  print(f"X shape: {X.shape}, Y shape: {Y.shape}")  

  # X_cropped = random_crop(X, crop_size=image_size)
  set_seed(209)
  X_cutmix, Y_cutmix = cutmix(X, Y, mask_size=mask_size)

  # X_cropped_no_indexing = random_crop_no_indexing(X, crop_size=image_size)

  # check if the results are the same
  # assert X_cropped.shape == X_cropped_no_indexing.shape, f"Cropped shapes do not match!, {X_cropped.shape} vs {X_cropped_no_indexing.shape}"
  # assert (X_cropped.numpy() == X_cropped_no_indexing.numpy()).all(), "Cropped results do not match!"
  # for x in (X_cropped, X_cropped_no_indexing):
  #   print(x[0:10, 0, 0, 0].numpy())  # Print first 10 values of the first image

  set_seed(209)
  X_cutmix_no_indexing, Y_cutmix_no_indexing = cutmix_no_indexing(X, Y, mask_size=mask_size)
  assert X_cutmix.shape == X_cutmix_no_indexing.shape, f"Cutmix X shapes do not match!, {X_cutmix.shape} vs {X_cutmix_no_indexing.shape}"
  assert Y_cutmix.shape == Y_cutmix_no_indexing.shape, f"Cutmix Y shapes do not match!, {Y_cutmix.shape} vs {Y_cutmix_no_indexing.shape}"
  for x in (X_cutmix, X_cutmix_no_indexing):
    print(x[0, 0:10, 0:10, 0].numpy())  # Print first 10 values of the first image

  # Number of unmatched pixels
  print(f"Unmatched in X: {((X_cutmix != X_cutmix_no_indexing).sum()).item()}")
  print(f"Unmatched in Y: {((Y_cutmix != Y_cutmix_no_indexing).sum()).item()}")
  # Number of pixels
  print(f"Total pixels in X: {X_cutmix.numel()}")
  print(f"Total pixels in Y: {Y_cutmix.numel()}")

  # Make an image to visualize the results

  separator = 10
  full_image = np.zeros((image_size * n_images, image_size * 3 + 2 * separator, 3))

  to_col = lambda x: (np.vstack(x.numpy().transpose(0, 2, 3, 1)))
  # import code; code.interact(local=locals())
  full_image[0:image_size*n_images, 0:image_size, :] = to_col(X)
  full_image[0:image_size*n_images, image_size + separator:image_size * 2 + separator, :] = to_col(X_cutmix)
  full_image[0:image_size*n_images, image_size * 2 + 2 * separator:image_size * 3 + 2 * separator, :] = to_col(X_cutmix_no_indexing)
  print(full_image.shape)

  img = Image.fromarray((full_image * 255).astype(np.uint8))
  AutoImage(img).draw()

  assert (X_cutmix.numpy() == X_cutmix_no_indexing.numpy()).all(), "Cutmix X results do not match!"
  print(Y_cutmix[0].numpy())
  print(Y_cutmix_no_indexing[0].numpy())
  assert (Y_cutmix.numpy() == Y_cutmix_no_indexing.numpy()).all(), "Cutmix Y results do not match!"