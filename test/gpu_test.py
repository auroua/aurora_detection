__author__ = 'auroua'

'''
test cudamat and gnumpy
'''

#!/usr/bin/python

import gnumpy
import numpy
from time import time

ox=[range(1000) for x in range(1000)]
oy=[range(1000) for y in range(1000)]

m=gnumpy.garray(ox)
n=gnumpy.garray(oy)

p=numpy.array(ox)
q=numpy.array(oy)

def run_gnumpy(a,b):
    st_g = time()
    gnumpy.dot(a,b)
    et_g = time()
    return et_g-st_g

def run_numpy(a,b):
    st_n = time()
    numpy.dot(a,b)
    et_n = time()
    return et_n-st_n

print (str(run_numpy(p,q)/run_gnumpy(m,n)))

