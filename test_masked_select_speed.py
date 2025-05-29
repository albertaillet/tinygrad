import time
from tinygrad import dtypes, Tensor
from tinygrad.helpers import getenv

def make_mask(shape, mask_size:int) -> Tensor:
  BS, W = shape
  low = Tensor.randint(BS, low=0, high=W-mask_size).reshape(BS,1)
  idx = Tensor.arange(W, dtype=dtypes.int32).reshape((1,W))
  return (idx >= low) * (idx < (low + mask_size))

def random_crop(X:Tensor, crop_size:int):
  mask = make_mask(X.shape, crop_size)
  X_cropped = Tensor(X.numpy()[mask.numpy()])
  return X_cropped.reshape((-1, crop_size))

def random_crop_in_tinygrad(X:Tensor, crop_size:int):
  mask = make_mask(X.shape, crop_size)
  return X.masked_select(mask).reshape((-1, crop_size))

def test_crop(X:Tensor, crop_size:int, seed:int):
  t1 = time.monotonic()
  Tensor.manual_seed(seed)
  X_cropped = random_crop(X, crop_size=crop_size).numpy()
  t2 = time.monotonic()
  Tensor.manual_seed(seed)
  X_cropped_in_tinygrad = random_crop_in_tinygrad(X, crop_size=crop_size).numpy()
  t3 = time.monotonic()
  assert (X_cropped == X_cropped_in_tinygrad).all(), "Cropped results do not match"
  t4 = time.monotonic()
  print(f"{(t2-t1)*1000.0:7.2f} ms numpy, {(t3-t2)*1000.0:7.2f} ms tinygrad, {(t4-t3)*1000.0:7.2f} ms comparison")

if __name__ == "__main__":
  BS, SEED = getenv("BS", 512), getenv("SEED", 42)
  Tensor.manual_seed(SEED)
  X = Tensor.rand((BS, 32), dtype=dtypes.float32)
  print(f"Batch size: {BS}, Seed: {SEED}")
  test_crop(X, crop_size=32, seed=SEED)
  print("Tests passed")