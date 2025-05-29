#!/usr/bin/env python
from tinygrad import Tensor

size = (4, 4)
a = Tensor.empty(*size)
b = Tensor.empty(*size)
print((a+b).tolist())