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

def random_crop_index(X:Tensor, pad_size:int):
  BS, W = X.shape
  low = Tensor.randint(BS, low=0, high=2*pad_size).reshape(BS,1)
  idx = Tensor.arange(W, dtype=dtypes.int32).reshape((1,W))
  idx = (idx - pad_size + low)  # start from padding and add offset
  idx = idx.abs() # left reflect
  idx = (idx < W).where(idx, 2*(W-1)-idx) # right reflect
  return X.gather(dim=1, index=idx)

def pad_reflect(X:Tensor, size:int) -> Tensor:
  return X[...,:,1:size+1].flip(-1).cat(X, X[...,:,-(size+1):-1].flip(-1), dim=-1)

def test_crop(X:Tensor, crop_size:int, seed:int, pad_size:int):
  X_padded = pad_reflect(X, size=pad_size)
  t1 = time.monotonic()
  Tensor.manual_seed(seed)
  X_cropped = random_crop(X_padded, crop_size=crop_size).numpy()
  t2 = time.monotonic()
  Tensor.manual_seed(seed)
  X_cropped_in_tinygrad = random_crop_in_tinygrad(X_padded, crop_size=crop_size).numpy()
  t3 = time.monotonic()
  Tensor.manual_seed(seed)
  X_cropped_index = random_crop_index(X, pad_size=pad_size).numpy()
  t4 = time.monotonic()
  for k,v in locals().items():
    if k.startswith("X"): print(f"{k:20}\n{v.numpy() if isinstance(v, Tensor) else v}\n")
  assert (X_cropped == X_cropped_in_tinygrad).all(), "Cropped results do not match"
  assert (X_cropped == X_cropped_index).all(), "Cropped results with index do not match"
  print(f"{(t2-t1)*1000.0:7.2f} ms numpy, {(t3-t2)*1000.0:7.2f} ms tinygrad, {(t4-t3)*1000.0:7.2f} ms index")

if __name__ == "__main__":
  SIZE, BS, SEED, PAD_SIZE = getenv("SIZE", 512), getenv("BS", 512), getenv("SEED", 42), getenv("PAD_SIZE", 2)
  Tensor.manual_seed(SEED)
  # X = Tensor.rand((BS, SIZE), dtype=dtypes.float32)
  X = Tensor.arange(SIZE, dtype=dtypes.float32).expand((BS, SIZE))
  print(f"Batch size: {BS}, Seed: {SEED}")
  test_crop(X, crop_size=SIZE, seed=SEED, pad_size=PAD_SIZE)
  print("Tests passed")