import time
from tinygrad import dtypes, Tensor
from tinygrad.nn import datasets
from tinygrad.helpers import getenv

def set_seed(seed):
  Tensor.manual_seed(seed)

def make_square_mask(shape, mask_size:int) -> Tensor:
  BS, _, H, W = shape
  low_x = Tensor.randint(BS, low=0, high=W-mask_size).reshape(BS,1,1,1)
  low_y = Tensor.randint(BS, low=0, high=H-mask_size).reshape(BS,1,1,1)
  idx_x = Tensor.arange(W, dtype=dtypes.int32).reshape((1,1,1,W))
  idx_y = Tensor.arange(H, dtype=dtypes.int32).reshape((1,1,H,1))
  return (idx_x >= low_x) * (idx_x < (low_x + mask_size)) * (idx_y >= low_y) * (idx_y < (low_y + mask_size))

def random_crop(X:Tensor, crop_size:int):
  mask = make_square_mask(X.shape, crop_size)
  mask = mask.expand((-1,3,-1,-1))
  X_cropped = Tensor(X.numpy()[mask.numpy()])
  return X_cropped.reshape((-1, 3, crop_size, crop_size))

def random_crop_in_tinygrad(X:Tensor, crop_size:int):
  mask = make_square_mask(X.shape, crop_size)
  return X.masked_select(mask).reshape((-1, 3, crop_size, crop_size))

def pad_reflect(X:Tensor, size:int) -> Tensor:
    X = X[...,:,1:size+1].flip(-1).cat(X, X[...,:,-(size+1):-1].flip(-1), dim=-1)
    X = X[...,1:size+1,:].flip(-2).cat(X, X[...,-(size+1):-1,:].flip(-2), dim=-2)
    return X

def test_crop(X:Tensor, crop_size:int, seed:int):
  t1 = time.monotonic()
  set_seed(seed)
  X_cropped = random_crop(X, crop_size=crop_size).numpy()
  t2 = time.monotonic()
  set_seed(seed)
  X_cropped_in_tinygrad = random_crop_in_tinygrad(X, crop_size=crop_size).numpy()
  t3 = time.monotonic()
  assert (X_cropped == X_cropped_in_tinygrad).all(), "Cropped results do not match"
  t4 = time.monotonic()
  print(f"{(t2-t1)*1000.0:7.2f} ms, {(t3-t2)*1000.0:7.2f} ms, {(t4-t3)*1000.0:7.2f} ms")

if __name__ == "__main__":
  BS, SEED = getenv("BS", 512), getenv("SEED", 42)
  X, _, _, _ = datasets.cifar()
  X = pad_reflect(X, size=2)  # pad to 36x36
  X = X[:BS]  # limit batch size
  print(f"Batch size: {BS}, Seed: {SEED}")
  print("Crop:")
  test_crop(X, crop_size=32, seed=SEED)
  print("Tests passed")