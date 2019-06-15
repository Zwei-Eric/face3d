import sys
import numpy as np
sys.path.append("../..")
from face3d import tensor


a = np.arange(24).reshape([3,4,2])
b = np.ones(4)
c = np.ones(2)

print(tensor.mode_dot.unfold(a, 0))
print(tensor.mode_dot.unfold(a, 1))
print(tensor.mode_dot.unfold(a, 2))
print(tensor.mode_dot.mode_dot(a,b,1))
