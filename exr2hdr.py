import glob
import os
import cv2

files = glob.glob('exr_format/Boitard/*.exr')
savepath = 'Boitard_HDR_format/'
for file in files:
    hdr = cv2.imread(file,-1)
    filename,file_ext = os.path.splitext(file)
    filename = os.path.basename(filename)
    filename = filename + '.hdr'
    curpath = os.path.join(savepath,filename)
    cv2.imwrite(curpath,hdr)

#---------------------------------------------------
# # Link: https://www.jianshu.com/p/f68637454079
# import numpy
# import OpenEXR
# import Imath
# import imageio
# import glob
# import os


# def ext2hdr(exrpath):
#     File = OpenEXR.InputFile(exrpath)
#     PixType = Imath.PixelType(Imath.PixelType.FLOAT)
#     DW = File.header()['dataWindow']
#     Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
#     rgb = [numpy.fromstring(File.channel(c, PixType), dtype=numpy.float32) for c in 'RGB']
#     r =numpy.reshape(rgb[0],(Size[1],Size[0]))
#     g =numpy.reshape(rgb[1],(Size[1],Size[0]))
#     b =numpy.reshape(rgb[2],(Size[1],Size[0]))
#     hdr = numpy.zeros((Size[1],Size[0],3),dtype=numpy.float32)
#     hdr[:,:,0] = r
#     hdr[:,:,1] = g
#     hdr[:,:,2] = b
#     return hdr

# def writehdr(hdrpath,hdr):
#     imageio.imwrite(hdrpath,hdr,format='hdr')


# files = glob.glob('./data/Fairchild/*.exr')
# savepath = './data/Fairchild/HDR_format'
# for file in files:
#     hdr = ext2hdr(file)
#     filename,file_ext = os.path.splitext(file)
#     filename = os.path.basename(filename)
#     filename = filename + '.hdr'
#     curpath = os.path.join(savepath,filename)
#     writehdr(curpath,hdr)