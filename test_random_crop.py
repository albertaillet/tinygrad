import time
from tinygrad import dtypes, Tensor
from tinygrad.nn import datasets
from tinygrad.helpers import getenv

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
  BS, C, H_padded, W_padded = X.shape
  H, W = H_padded - 2*pad_size, W_padded - 2*pad_size
  low_x = Tensor.randint(BS, low=0, high=2*pad_size).reshape(BS,1,1,1)
  low_y = Tensor.randint(BS, low=0, high=2*pad_size).reshape(BS,1,1,1)
  idx_x = Tensor.arange(W, dtype=dtypes.int32).reshape(1,1,1,W)
  idx_y = Tensor.arange(H, dtype=dtypes.int32).reshape(1,1,H,1)
  mask_idx_x = (idx_x + low_x).expand(BS, C, H_padded, W)
  mask_idx_y = (idx_y + low_y).expand(BS, C, H, W)
  return X.gather(3, mask_idx_x).gather(2, mask_idx_y)

def pad_reflect(X:Tensor, size:int) -> Tensor:
  X = X[...,:,1:size+1].flip(-1).cat(X, X[...,:,-(size+1):-1].flip(-1), dim=-1)
  X = X[...,1:size+1,:].flip(-2).cat(X, X[...,-(size+1):-1,:].flip(-2), dim=-2)
  return X

def test_crop(X:Tensor, crop_size:int, seed:int, pad_size:int):
  X_padded = pad_reflect(X, size=pad_size)
  t1 = time.monotonic()
  Tensor.manual_seed(seed)
  X_cropped_numpy = random_crop(X_padded, crop_size=crop_size).numpy()
  t2 = time.monotonic()
  Tensor.manual_seed(seed)
  X_cropped_index = random_crop_index(X_padded, pad_size=pad_size).numpy()
  t3 = time.monotonic()
  Tensor.manual_seed(seed)
  if BS < MASKED_SELECT_MAX_BS: X_cropped_masked_select = random_crop_masked_select(X_padded, crop_size=crop_size).numpy()
  t4 = time.monotonic()
  assert (X_cropped_numpy == X_cropped_index).all()
  if BS < MASKED_SELECT_MAX_BS: assert (X_cropped_numpy == X_cropped_masked_select).all()
  print(f"Dataset shape: {str(X.shape):>18}, {(t2-t1)*1000.0:7.2f} ms numpy, {(t3-t2)*1000.0:7.2f} ms index, {(t4-t3)*1000.0:7.2f} ms masked_select")

if __name__ == "__main__":
  SEED, PAD_SIZE, MASKED_SELECT_MAX_BS = getenv("SEED", 42), getenv("PAD_SIZE", 2), getenv("MASKED_SELECT_MAX_BS", 1000)
  X, _, _, _ = datasets.cifar()
  for BS in [100, 500, 50000]:
    test_crop(X[:BS], crop_size=32, seed=SEED, pad_size=PAD_SIZE)