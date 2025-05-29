#!/usr/bin/env python
from tinygrad.shape.view import View

# The two by two matrix is represneted as shape being (2, 2), strides being (2, 1)
a = View.create(shape=(2, 2), strides=(2, 1))

# In order to get the "strategy" we talked about (row * 2 + col), we call a special method:
idx, valid = a.to_indexed_uops()

# This method returns two objects. The first one is the access strategy I kept referring to, the second is for the mask.
# Let's focus on the first.
# print(idx)

# If you print it, it may look a bit scary:
# UOp(Ops.ADD, dtypes.int, arg=None, src=(
#   UOp(Ops.ADD, dtypes.int, arg=None, src=(
#     x1:=UOp(Ops.CONST, dtypes.int, arg=0, src=()),
#     UOp(Ops.MUL, dtypes.int, arg=None, src=(
#       UOp(Ops.RANGE, dtypes.int, arg=0, src=(
#          x1,
#         UOp(Ops.CONST, dtypes.int, arg=3, src=()),)),
#       x5:=UOp(Ops.CONST, dtypes.int, arg=2, src=()),)),)),
#   UOp(Ops.MUL, dtypes.int, arg=None, src=(
#     UOp(Ops.RANGE, dtypes.int, arg=1, src=(
#        x1,
#        x5,)),
#     UOp(Ops.CONST, dtypes.int, arg=1, src=()),)),))
# Fortunately, there's a render method to show you something more sane:
print(idx.render())