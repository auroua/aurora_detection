__author__ = 'auroua'
import numpy as np
import cudamat as cm

cm.cublas_init()

# create two random matrices and copy them to the GPU
a = cm.CUDAMatrix(np.random.rand(32, 256))
b = cm.CUDAMatrix(np.random.rand(256, 32))
c = cm.CUDAMatrix(np.random.rand(32, 256))
c1 = cm.CUDAMatrix(np.random.rand(32, 256))
e = cm.CUDAMatrix(np.random.rand(32, 256))
# perform calculations on the GPU
c = cm.dot(a, b)
print b.shape
d = c.sum(axis = 0)
a.add(c1, target=a)

# copy d back to the host (CPU) and print
print(d.asarray())
print(a.asarray())