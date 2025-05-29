import random
from tinygrad import dtypes, Tensor
from tinygrad.helpers import getenv
from tinygrad.nn import datasets

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

# ========== Previous version ==========

def random_crop(X:Tensor, crop_size=32):
  mask = make_square_mask(X.shape, crop_size)
  mask = mask.expand((-1,3,-1,-1))
  X_cropped = Tensor(X.numpy()[mask.numpy()])
  return X_cropped.reshape((-1, 3, crop_size, crop_size))

def cutmix(X:Tensor, Y:Tensor, mask_size=3):
  # fill the square with randomly selected images from the same batch
  mask = make_square_mask(X.shape, mask_size)
  order = Tensor.randperm(X.shape[0], device=X.device, dtype=dtypes.int32).numpy()
  X_patch = Tensor(X.numpy()[order], device=X.device, dtype=X.dtype)
  Y_patch = Tensor(Y.numpy()[order], device=Y.device, dtype=Y.dtype)
  X_cutmix = mask.where(X_patch, X)
  mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
  Y_cutmix = mix_portion * Y_patch + (1. - mix_portion) * Y
  return X_cutmix, Y_cutmix

# ========== In tinygrad ==========

def random_crop_in_tinygrad(X:Tensor, crop_size=32):

  mask = make_square_mask(X.shape, crop_size)
  print(mask.realize(), X.shape)
  # return X.masked_select(mask).reshape((-1, 3, crop_size, crop_size))
  print(mask)
  return X[mask].reshape((-1, 3, crop_size, crop_size))


def cutmix_in_tinygrad(X:Tensor, Y:Tensor, mask_size=3):
  # fill the square with randomly selected images from the same batch
  mask = make_square_mask(X.shape, mask_size)
  order = Tensor.randperm(X.shape[0], device=X.device, dtype=dtypes.int32)
  X_patch, Y_patch = X[order], Y[order]
  X_cutmix = mask.where(X_patch, X)
  mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
  Y_cutmix = mix_portion * Y_patch + (1. - mix_portion) * Y
  return X_cutmix, Y_cutmix

if __name__ == "__main__":
  BS, SEED = getenv("BS", 512), getenv("SEED", 42)
  X, Y, _, _ = datasets.cifar()
  X, Y = X[:BS], Y[:BS].one_hot(10)

  set_seed(SEED)
  X_cropped = random_crop(X)
  X_cutmix, Y_cutmix = cutmix(X, Y)

  set_seed(SEED)
  X_cropped_in_tinygrad = random_crop_in_tinygrad(X)
  X_cutmix_in_tinygrad, Y_cutmix_in_tinygrad = cutmix_in_tinygrad(X, Y)

  assert (X_cropped == X_cropped_in_tinygrad).numpy().all(), "Cropped results do not match"
  assert (X_cutmix == X_cutmix_in_tinygrad).numpy().all(), "Cutmix X results do not match"
  assert (Y_cutmix == Y_cutmix_in_tinygrad).numpy().all(), "Cutmix Y results do not match"
  print("Tests passed")