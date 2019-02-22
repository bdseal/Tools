作者：学而时习之_不亦说乎
链接：https://www.jianshu.com/p/9ec56af503c1


## Python进行TIFF文件处理
很多医学文件采用格式TIFF格式存储，并且一个TIFF文件由多帧序列组合而成，使用libtiff可以将TIFF文件中的多帧提取出来。

from libtiff import TIFF

def tiff2Stack(filePath):
    tif = TIFF.open(filePath,mode='r')
    stack = []
    for img in list(tif.iter_images()):
        stack.append(img)
    return  stack


## Read HDR in python
因为希望利用神经网络处理HDR图像，所以首先要用python读入HDR文件。 还好在Python中有一个imageio库可以对各种图像进行读取，利用pip来安装：

pip install imageio
imageio依赖一个叫做freeimage的库，在ubuntu系统下的安装也十分方便

sudo apt-get install libfreeimage3 libfreeimage-dev
下面我们使用Python来读取一个HDR图像：

import imageio
import matplotlib.pyplot as plt
img = imageio.imread("PATH-TO-HDR-FILE/memorial.hdr")
%matplotlib inline
plt.imshow(img);


## Python进行高动态图像合成
利用几个helper函数来从文件夹里读取图像和图像的曝光时间。

import PIL.ExifTags
from PIL import Image
import cv2
import numpy as np
from libtiff import TIFF
from os import listdir
from os.path import isfile, isdir, join

#读取文件夹下文件
def ListFiles(FilePath):
    onlyfiles = [f for f in listdir(FilePath) if isfile(join(FilePath, f))]
    return onlyfiles

#获得图像文件属性
def get_exif(fn):
    img = Image.open(fn)
    exif = {PIL.ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in PIL.ExifTags.TAGS
            }
    return exif

#获得图像曝光时间
def get_exposure_time(fn):
    exif = get_exif(fn)
    exposure_time = exif.get('ExposureTime')
    return exposure_time[0]/exposure_time[1]

#获取图像曝光时间序列和图像
def getImageStackAndExpos(folderPath):
    files = ListFiles(folderPath)
    exposTimes = []
    imageStack = []
    for file in files:
        filePath = join(folderPath,file)
        exposTime = get_exposure_time(filePath)
        currImage = cv2.imread(filePath)
        exposTimes.append(exposTime)
        imageStack.append(currImage)
    #根据曝光时间长短，对图像序列和曝光时间序列重新排序
    index = sorted(range(len(exposTimes)), key=lambda k: exposTimes[k])
    exposTimes = [exposTimes[i] for i in index]
    imageStack = [imageStack[i] for i in index]
    return exposTimes,imageStack
预处理
由于上面的三张图像在拍摄的时候存在一定的抖动和位移情况，可以用SIFT算法对其配准Python进行SIFT图像对准。在配准的时候，首先需要找到一张曝光最好的照片作为基准照片，下面的一个函数计算照片中曝光不足和曝光过量的像素个数，把曝光不足和曝光过量像素最少的图像作为参考图像。

def getSaturNum(img):
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    underExpos=np.count_nonzero(gray_image==0)
    overExpos = np.count_nonzero(gray_image==255)
    return underExpos + overExpos


def getRefImage(imgStack):
    saturNum = imgStack[0].shape[0]*imgStack[0].shape[1]
    for imgIndex in np.arange(len(imgStack)):
        curImg = imgStack[imgIndex]
        curSaturNum = getSaturNum(curImg)
        print(curSaturNum)
        if curSaturNum <= saturNum:
            saturNum = curSaturNum
            refIndex = imgIndex

    return  refIndex
在获得参考图像以后，使用Python进行SIFT图像对准中提供的siftImageAlignment进行配准，并且返回已经对其的图像序列。

def siftAlignment(imgStack,refIndex):
    refImg = imgStack[refIndex]
    outStack = []
    for index in np.arange(len(imgStack)):
        if index == refIndex:
            outStack.append(refImg)
        else:
            currImg = imgStack[index]
            outImg,_,_ = siftImageAlignment(refImg,currImg)
            outStack.append(outImg)
    return outStack
HDR合成
为了合成HDR图像，首先要拟合相机响应函数（Camera Response Function，CRF），关于拟合CRF的算法，后续博客中将详细介绍，这里，我们先关注Opencv-Python的实现。

import numpy as np
import cv2
import Utility #Utility为前面函数所在的模块

exposTimes,images = Utility.getImageStackAndExpos('stack_alignment')
refImgIndex= Utility.getRefImage(images)
images = Utility.siftAlignment(images,refImgIndex) 

exposTimes = np.array(exposTimes,dtype=np.float32) #需要转化为numpy浮点数组
calibrateDebevec = cv2.createCalibrateDebevec(samples=120,random=True)  
###采样点数120个，采样方式为随机，一般而言，采用点数越多，采样方式越随机，最后的CRF曲线会越加平滑
responseDebevec = calibrateDebevec.process(images, exposTimes)  #获得CRF
mergeDebevec = cv2.createMergeDebevec()
hdrDebevec = mergeDebevec.process(images, exposTimes, responseDebevec) #
# Save HDR image.
cv2.imwrite("hdrDebevec.hdr", hdrDebevec)

作者：学而时习之_不亦说乎
链接：https://www.jianshu.com/p/edb29842001a
來源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

