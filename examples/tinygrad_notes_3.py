#!/usr/bin/env python
from tinygrad import dtypes
from tinygrad.ops import Ops, UOp
from tinygrad.renderer.cstyle import MetalRenderer

metal_renderer = MetalRenderer()
const = UOp(Ops.CONST, dtypes.float, arg=1.0)
define_global = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0)
special = UOp(Ops.SPECIAL, dtypes.int, arg=('gidx0', 16), src=())
added = UOp(Ops.ADD, dtypes.long, arg=None, src=(define_global, special))
store = UOp(Ops.STORE, dtypes.void, arg=None, src=(added, const))
uops = [const, define_global, special, added, store]

rendered = metal_renderer.render(uops)
print(rendered)

"""
#include <metal_stdlib>
using namespace metal;
kernel void rendered(device float* data0, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 16 */
  *(data0+gidx0) = 1.0f;
}
"""