#!/usr/bin/env python
from tinygrad.renderer.cstyle import MetalRenderer, CUDARenderer
from tinygrad.ops import UOp, Ops
from tinygrad import dtypes

const = UOp(Ops.CONST, dtypes.float, arg=1.0)
add = UOp(Ops.ADD, dtypes.float, src=(const, const), arg=None)

# print(add)
# print("\n\n")
# print(MetalRenderer().render([const, add]))

print(MetalRenderer().render([UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", 16))]))

print(CUDARenderer("sm_50").render([
  UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", 16)),
  UOp(Ops.SPECIAL, dtypes.int, arg=("gidx1", 16))
]))