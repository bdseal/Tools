# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 20:58:39 2018

@author: Will Hsia
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy  as np
import scipy.ndimage.filters as fi
import scipy.stats as st
def gaussian_2d_kernel(kernel_size = 7,sigma = 0):

    kernel = np.zeros([kernel_size,kernel_size])
    center = kernel_size//2

    if sigma == 0:
        sigma = ((kernel_size-1)*0.5 - 1)*0.3 + 0.8

    s = 2*(sigma**2)
    sum_val = 0
    for i in range(0,kernel_size):
        for j in range(0,kernel_size):
            x = i-center
            y = j-center
            kernel[i,j] = np.exp(-(x**2+y**2) / s)
            sum_val += kernel[i,j]
            #/(np.pi * s)
    sum_val = 1/sum_val
    return kernel*sum_val
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel
def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)
ke=gaussian_2d_kernel(7,2)
ke2=gkern2(17,20)
np.set_printoptions(threshold='nan')
print(ke2)