from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy
import numpy.linalg as la
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void subtour(int *path, int path_sz, float *city_x, float *city_y)
{
    const int MAX_PATH_SZ = 100;
    if (path_sz <= MAX_PATH_SZ) {
        // make sure city_x, city_y is copied to shared mem
        __shared__ float *s_city_x[MAX_PATH_SZ];
        __shared__ float *s_city_y[MAX_PATH_SZ];

        // Lin Kernighan 
        
    
    }
}
""")

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

dest = numpy.zeros_like(a)
multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400,1,1))

print(dest-a*b)
