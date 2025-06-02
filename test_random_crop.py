import time
from tinygrad import dtypes, Tensor
from tinygrad.nn import datasets
from tinygrad.helpers import getenv

def set_seed(seed):
  Tensor.manual_seed(seed)

def random_low_and_indices(batch_size:int, dim_size:int, high:int):
  low = Tensor.randint(batch_size, low=0, high=high).reshape(batch_size,1)
  idx = Tensor.arange(dim_size, dtype=dtypes.int32).reshape(1,dim_size)
  return low, idx

def make_square_mask(shape, mask_size:int) -> Tensor:
  BS, _, H, W = shape
  low_x = Tensor.randint(BS, low=0, high=W-mask_size).reshape(BS,1,1,1)
  low_y = Tensor.randint(BS, low=0, high=H-mask_size).reshape(BS,1,1,1)
  idx_x = Tensor.arange(W, dtype=dtypes.int32).reshape((1,1,1,W))
  idx_y = Tensor.arange(H, dtype=dtypes.int32).reshape((1,1,H,1))
  return (idx_x >= low_x) * (idx_x < (low_x + mask_size)) * (idx_y >= low_y) * (idx_y < (low_y + mask_size))

def random_crop(X:Tensor, crop_size:int):
  mask = make_square_mask(X.shape, crop_size)
  mask = mask.expand(-1,3,-1,-1)
  X_cropped = Tensor(X.numpy()[mask.numpy()])
  return X_cropped.reshape(-1, 3, crop_size, crop_size)

def random_crop_masked_select(X:Tensor, crop_size:int):
  mask = make_square_mask(X.shape, crop_size)
  return X.masked_select(mask).reshape((-1, 3, crop_size, crop_size))

def random_crop_index(X:Tensor, pad_size:int):
  BS, C, H_pad, W_pad = X.shape
  H, W = H_pad - 2*pad_size, W_pad - 2*pad_size
  low_x, idx_x = random_low_and_indices(BS, W, 2*pad_size)
  low_y, idx_y = random_low_and_indices(BS, H, 2*pad_size)
  idx_x = (idx_x+low_x).reshape(BS,1,W)
  idx_y = (idx_y+low_y).reshape(BS,H,1)
  idx_flat = (idx_y * W_pad + idx_x).reshape(BS, 1, H*W).expand(BS, C, H*W)
  X_flat = X.reshape(BS, C, H_pad*W_pad)
  return X_flat.gather(2, idx_flat).reshape(BS, C, H, W)

def pad_reflect(X:Tensor, size:int) -> Tensor:
  X = X[...,:,1:size+1].flip(-1).cat(X, X[...,:,-(size+1):-1].flip(-1), dim=-1)
  X = X[...,1:size+1,:].flip(-2).cat(X, X[...,-(size+1):-1,:].flip(-2), dim=-2)
  return X

def test_crop(X:Tensor, crop_size:int, seed:int, pad_size:int):
  X_padded = pad_reflect(X, size=pad_size)
  t1 = time.monotonic()
  set_seed(seed)
  X_cropped_numpy = random_crop(X_padded, crop_size=crop_size).numpy()
  t2 = time.monotonic()
  set_seed(seed)
  X_cropped_index = random_crop_index(X_padded, pad_size=pad_size).numpy()
  t3 = time.monotonic()
  set_seed(seed)
  if X_padded.shape[0] < 5000:
    X_cropped_masked_select = random_crop_masked_select(X_padded, crop_size=crop_size).numpy()
  t4 = time.monotonic()
  for k,v in locals().items():
    if k.startswith("X") and PRINT: print(f"{k:20}\n{v.numpy() if isinstance(v, Tensor) else v}\n")
  assert (X_cropped_numpy == X_cropped_index).all(), "Cropped results with index do not match"
  if X.shape[0] < 5000:
    assert (X_cropped_numpy == X_cropped_masked_select).all(), "Cropped results do not match"
  print(f"Dataset shape: {str(X_padded.shape):>18}, {(t2-t1)*1000.0:7.2f} ms numpy, {(t3-t2)*1000.0:7.2f} ms index, {(t4-t3)*1000.0:7.2f} ms masked_select")

if __name__ == "__main__":
  SIZE, PAD_SIZE = getenv("SIZE", 32), getenv("PAD_SIZE", 2)
  SEED, CIFAR, PRINT = getenv("SEED", 42), getenv("CIFAR", 1), getenv("PRINT", 0)
  for BS in [100, 500, 50000]:
    if CIFAR:
      X, _, _, _ = datasets.cifar()
    else:
      X = Tensor.arange(BS * 3 * SIZE * SIZE, dtype=dtypes.int32).reshape(BS, 3, SIZE, SIZE)
    X = X[:BS, :3, :SIZE, :SIZE]
    test_crop(X, crop_size=SIZE, seed=SEED, pad_size=PAD_SIZE)