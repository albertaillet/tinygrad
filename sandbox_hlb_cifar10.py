#!/usr/bin/env python3

# tinygrad implementation of https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/
# https://siboehm.com/articles/22/CUDA-MMM
import random, time
from tinygrad import nn, dtypes, Tensor, Device, GlobalCounters, TinyJit
from tinygrad.helpers import getenv
from extra.bench_log import BenchEvent, WallTimeEvent

cifar_mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
cifar_std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

BS, STEPS = getenv("BS", 512), getenv("STEPS", 1000)
bias_scaler = 58
hyp = {
  'seed' : 209,
  'net': {
      'kernel_size': 2,             # kernel size for the whitening layer
      'cutmix_size': 3,
      'cutmix_steps': 499,
      'pad_amount': 2
  },
}

def train_cifar():

  def set_seed(seed):
    Tensor.manual_seed(seed)
    random.seed(seed)

  # ========== Preprocessing ==========
  # NOTE: this only works for RGB in format of NxCxHxW and pads the HxW
  def pad_reflect(X, size=2) -> Tensor:
    X = X[...,:,1:size+1].flip(-1).cat(X, X[...,:,-(size+1):-1].flip(-1), dim=-1)
    X = X[...,1:size+1,:].flip(-2).cat(X, X[...,-(size+1):-1,:].flip(-2), dim=-2)
    return X

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
    order = list(range(0, X.shape[0]))
    random.shuffle(order)
    X_patch = Tensor(X.numpy()[order], device=X.device, dtype=X.dtype)
    Y_patch = Tensor(Y.numpy()[order], device=Y.device, dtype=Y.dtype)
    X_cutmix = mask.where(X_patch, X)
    mix_portion = float(mask_size**2)/(X.shape[-2]*X.shape[-1])
    Y_cutmix = mix_portion * Y_patch + (1. - mix_portion) * Y
    return X_cutmix, Y_cutmix

  # the operations that remain inside batch fetcher is the ones that involves random operations
  def fetch_batches(X_in:Tensor, Y_in:Tensor, BS:int, is_train:bool):
    step, epoch = 0, 0
    while True:
      st = time.monotonic()
      X, Y = X_in, Y_in
      if is_train:
        # TODO: these are not jitted
        if getenv("RANDOM_CROP", 1):
          X = random_crop(X, crop_size=32)
        if getenv("RANDOM_FLIP", 1):
          X = (Tensor.rand(X.shape[0],1,1,1) < 0.5).where(X.flip(-1), X) # flip LR
        if getenv("CUTMIX", 1):
          if step >= hyp['net']['cutmix_steps']:
            X, Y = cutmix(X, Y, mask_size=hyp['net']['cutmix_size'])
        order = list(range(0, X.shape[0]))
        random.shuffle(order)
        X, Y = X.numpy()[order], Y.numpy()[order]
      else:
        X, Y = X.numpy(), Y.numpy()
      et = time.monotonic()
      print(f"shuffling {'training' if is_train else 'test'} dataset in {(et-st)*1e3:.2f} ms ({epoch=})")
      for i in range(0, X.shape[0], BS):
        # pad the last batch  # TODO: not correct for test
        batch_end = min(i+BS, Y.shape[0])
        x = Tensor(X[batch_end-BS:batch_end], device=X_in.device, dtype=X_in.dtype)
        y = Tensor(Y[batch_end-BS:batch_end], device=Y_in.device, dtype=Y_in.dtype)
        step += 1
        yield x, y
      epoch += 1
      if not is_train: break

  transform = [
    lambda x: x.float() / 255.0,
    lambda x: x.reshape((-1,3,32,32)) - Tensor(cifar_mean, device=x.device, dtype=x.dtype).reshape((1,3,1,1)),
    lambda x: x / Tensor(cifar_std, device=x.device, dtype=x.dtype).reshape((1,3,1,1)),
  ]
  set_seed(getenv('SEED', hyp['seed']))

  X_train, Y_train, X_test, Y_test = nn.datasets.cifar()
  # one-hot encode labels
  Y_train, Y_test = Y_train.one_hot(10), Y_test.one_hot(10)
  # preprocess data
  X_train, X_test = X_train.sequential(transform), X_test.sequential(transform)


  # padding is not timed in the original repo since it can be done all at once
  X_train = pad_reflect(X_train, size=hyp['net']['pad_amount'])

  # Convert data and labels to the default dtype
  X_train, Y_train = X_train.cast(dtypes.default_float), Y_train.cast(dtypes.default_float)
  X_test, Y_test = X_test.cast(dtypes.default_float), Y_test.cast(dtypes.default_float)

  i = 0
  eval_acc_pct = 0.0
  batcher = fetch_batches(X_train, Y_train, BS=BS, is_train=True)
  with Tensor.train():
    st = time.monotonic()
    while i <= STEPS:
      with WallTimeEvent(BenchEvent.STEP):
        X, Y = next(batcher)
        et = time.monotonic()

      cl = time.monotonic()
      device_str = X.device if isinstance(X.device, str) else f"{X.device[0]} * {len(X.device)}"
      print(f"{i:3d} {(cl-st)*1000.0:7.2f} ms run, {(et-st)*1000.0:7.2f} ms python, {(cl-et)*1000.0:7.2f} ms {device_str}, {GlobalCounters.mem_used/1e9:.2f} GB used, {GlobalCounters.global_ops*1e-9/(cl-st):9.2f} GFLOPS, {GlobalCounters.global_ops*1e-9:9.2f} GOPS")
      i += 1
if __name__ == "__main__":
  with WallTimeEvent(BenchEvent.FULL):
    train_cifar()
