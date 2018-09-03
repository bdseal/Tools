"""
 "
 " Description: create_LDRs, create multi-exposure ldr images or bracketed images via HDR.
 " refer to HDRCNN, based on U-net but different on skip connection.
 " Author: Will Hsia
 " Date: Feb 2018
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pylab as pl
import linecache  
from scipy import misc,interpolate

def CRF(x):
    "Please modify this function according to your CRFs"
    "Gamma Correction can be used to fit one CRF curve."
    return x**2.2

def getCRF(dorfCurves_FILE,Nth):
    I = linecache.getline(dorfCurves_FILE, 6*Nth+4) 
    B = linecache.getline(dorfCurves_FILE, 6*Nth+6)
    name_I = [float(x) for x in I.split( )]
    name_B = [float(x) for x in B.split( )]
#    return name_I.split( ),name_B.split( )
    return np.asarray(name_I),np.asarray(name_B)
    
def createLDRs_(input_path, output_path,dorfCurves_FILE,Nth):
    taus = [np.math.sqrt(2.)**x for x in range(-6,5)]
    #if you have opencv3, if not use readHDR instead.
    img = cv2.imread(input_path, -1) 
    print (img)
#    img=readHDR(input_path)
    img = 0.5 * img / img.mean()

    name_I,name_B=getCRF(dorfCurves_FILE,Nth)
    f = interpolate.interp1d(name_I,name_B, 'quadratic')
    tau = np.math.sqrt(2.)**(-7)
    ynew=255.0*f(np.clip(tau*img, 0., 1.0))
    out_ldr = np.clip(ynew, 0., 255.).astype('uint8')
    for i, tau in enumerate(taus):
#        out = np.clip((255.*CRF(tau*img)),0., 255.).astype('uint8')      
         ynew=255.0*f(np.clip(tau*img, 0., 1.0))
         out = np.clip(ynew, 0., 255.0).astype('uint8')
#         cv2.imwrite(output_path.rsplit('.',1)[0]+'_'+str(i)+'.'+output_path.rsplit('.',1)[-1], out)
         out_ldr=np.concatenate((out_ldr, out),axis=2)
    return out_ldr
    
def createLDRs(input_hdr,dorfCurves_FILE,Nth):
    taus = [np.math.sqrt(2.)**x for x in range(-5,3)]
    # taus = [np.math.sqrt(2.)**x for x in [-11,-8,-6,-4,-2,-1,0,1,2]]
    #if you have opencv3, if not use readHDR instead.
    img = cv2.imread(input_path, -1) 
#    img =input_hdr
#    img=readHDR(input_path)
    img = 0.5 * img / img.mean()
    name_I,name_B=getCRF(dorfCurves_FILE,Nth)
    f = interpolate.interp1d(name_I,name_B, 'quadratic')

    for i, tau in enumerate(taus):
         ynew=255.0*f(np.clip(tau*img, 0., 1.0))
         out = np.clip(ynew, 0., 255.0).astype('uint8')
         # output_path = "./output.png"
         # cv2.imwrite(output_path.rsplit('.',1)[0]+'_'+str(i)+'.'+output_path.rsplit('.',1)[-1], out)
         cv2.imwrite(str(i)+'.tif', out)
    return out

if __name__ == '__main__':
    dorfCurves_FILE='./dorfCurves.txt'
    Nth=2#[0,200]
    input_path = "/home/xwh/SITMO/sitmo_DeepHDR/dataset/demo/507/507.hdr"
#    input_path = "/media/sensetime/My Passport/git/hdr_training/dml/001.hdr"
    # input_path = "/media/sensetime/My Passport/git//sitmo_crop/sample_testing/094.hdr"
    # output_path = "/media/sensetime/My Passport/git/sitmo_crop/sample_testing/094/"
    out_ldr=createLDRs(input_path,dorfCurves_FILE,Nth)