#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:11:47 2018

@author: xwh
"""
import tensorflow as tf
import cv2
import numpy as np
import scipy.stats as st
# Gaussian Filter
#K = np.array([[0.003765,0.015019,0.023792,0.015019,0.003765],
#[0.015019,0.059912,0.094907,0.059912,0.015019],
#[0.023792,0.094907,0.150342,0.094907,0.023792],
#[0.015019,0.059912,0.094907,0.059912,0.015019],
#[0.003765,0.015019,0.023792,0.015019,0.003765]], dtype='float32')
#
#K = np.array([[0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067],  
#        [0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292],  
#        [0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117],  
#        [0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771],  
#        [0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117],  
#        [0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292],  
#        [0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067]] , dtype='float32')
K1 = np.array([[0.0049,    0.0092,    0.0134 ,   0.0152 ,   0.0134   , 0.0092  ,  0.0049], 
    [0.0092  ,  0.0172 ,   0.0250, 0.0283  ,  0.0250  ,  0.0172   , 0.0092],
    [0.0134   , 0.0250  ,  0.0364,    0.0412   , 0.0364   , 0.0250  ,  0.0134],
    [0.0152    ,0.0283   , 0.0412,    0.0467   , 0.0412  ,  0.0283  ,  0.0152],
    [0.0134 ,   0.0250   , 0.0364,    0.0412 ,   0.0364  ,  0.0250  ,  0.0134],
    [0.0092  ,  0.0172   , 0.0250,    0.0283 ,   0.0250  ,  0.0172  ,  0.0092],
    [0.0049   , 0.0092   , 0.0134,    0.0152 ,   0.0134  ,  0.0092  ,  0.0049]] , dtype='float32')
K2 = np.array([[0.0202   , 0.0203,    0.0204 ,   0.0204  ,  0.0204   , 0.0203  ,  0.0202], 
    [0.0203,    0.0204   , 0.0205,    0.0205 ,   0.0205  ,  0.0204  ,  0.0203], 
    [0.0204 ,   0.0205   , 0.0206,    0.0206 ,   0.0206  ,  0.0205  ,  0.0204], 
    [0.0204  ,  0.0205   , 0.0206 ,   0.0206 ,   0.0206  ,  0.0205  ,  0.0204], 
    [0.0204   , 0.0205   , 0.0206 ,   0.0206 ,   0.0206  ,  0.0205  ,  0.0204], 
    [0.0203    ,0.0204   , 0.0205 ,   0.0205 ,   0.0205  ,  0.0204  ,  0.0203], 
    [0.0202    ,0.0203   , 0.0204 ,   0.0204 ,   0.0204  ,  0.0203  ,  0.0202]] , dtype='float32')


#-------------------gauss_kernel test--------------
def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter
kernel_var = gauss_kernel(21, 3, 3); 
#---------------------------------
#img = cv2.imread('/home/xwh/git/DPED/dped/iphone/test_data/full_size_test_images/0.jpg')
#img = cv2.imread('/home/xwh/git/DPED_ICIP/dped/canon/40.jpg')
img = cv2.imread('/home/xwh/git/DPED/dped/iphone/test_data/patches/iphone/41.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

float_gray = gray.astype(np.float32) / 255.0
# as tensorflow constants with correct shapes
x = tf.constant(float_gray.reshape(1,float_gray.shape[0],float_gray.shape[1], 1))
w1 = tf.constant(K1.reshape(K1.shape[0],K1.shape[1], 1, 1))
w2 = tf.constant(K2.reshape(K2.shape[0],K2.shape[1], 1, 1))
with tf.Session() as sess:
    # get low/high pass ops
    lowpass = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
    num = x-lowpass    
    blur = tf.nn.conv2d(num*num, w2, strides = [1, 1, 1, 1], padding='SAME')
    den = tf.pow(blur, 0.5)
    gray = num / den
    gray_norm=(gray-tf.reduce_min(gray))/(tf.reduce_max(gray)-tf.reduce_min(gray))
#    gray.norm=tf.reshape(gray_norm,[-1,2432, 3648,1])
    gray.norm=tf.reshape(gray_norm,[-1,100, 100,1])
    # get high pass image
    l = sess.run(gray)
    l2 = sess.run(gray_norm)
    
    l = l.reshape(img.shape[0],img.shape[1])
#    l2 = l2.reshape(img.shape[0],img.shape[1])
    cv2.normalize(l, dst=l, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite("./debug.png", l * 255)
    cv2.imwrite("./debug_canon_tf_softmax.png", l2 * 255)

 


