#!/usr/bin/env python
from tinygrad import dtypes
from tinygrad.ops import Ops, UOp
from tinygrad.renderer.cstyle import MetalRenderer

from tinygrad.ops import PatternMatcher, UPat

const_1 = UOp(Ops.CONST, dtypes.float, arg=0.5)
const_2 = UOp(Ops.CONST, dtypes.float, arg=0.5)

matcher = PatternMatcher([
  (UPat(Ops.CONST, dtypes.float, name="x"), lambda ctx, x: UOp(Ops.ADD, dtypes.float, src=(const_1, const_2))),
])

metal_renderer = MetalRenderer()
const = UOp(Ops.CONST, dtypes.float, arg=1.0)

const_rewritten = matcher.rewrite(const)
print(const_rewritten)

exit(0)
define_global = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0)
special = UOp(Ops.SPECIAL, dtypes.int, arg=('gidx0', 16), src=())
added = UOp(Ops.ADD, dtypes.long, arg=None, src=(define_global, special))
store = UOp(Ops.STORE, dtypes.void, arg=None, src=(added, const_rewritten))
uops = [const_1, const_2, const_rewritten, define_global, special, added, store]

rendered = metal_renderer.render(uops)
print(rendered)
"""
#include <metal_stdlib>
using namespace metal;
kernel void rendered(device float* data0, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 16 */
  *(data0+gidx0) = (0.5f+0.5f);
}
"""