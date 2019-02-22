#https://stackoverflow.com/questions/43240604/python-local-normalization-in-opencv
##https://stackoverflow.com/questions/43279519/implementing-high-pass-filter-in-tensorflow
import cv2
import numpy as np
ks=17
img = cv2.imread('/home/xwh/git/DPED/dped/iphone/test_data/full_size_test_images/0.jpg')
img = cv2.imread('/home/xwh/git/DPED_ICIP/dped/canon/40.jpg')
img = cv2.imread('/home/xwh/git/DPED/dped/iphone/test_data/patches/canon/41.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

float_gray = gray.astype(np.float32) / 255.0

blur = cv2.GaussianBlur(float_gray, (ks, ks), sigmaX=2, sigmaY=2)
num = float_gray - blur

blur = cv2.GaussianBlur(num*num, (ks, ks), sigmaX=20, sigmaY=20)
den = cv2.pow(blur, 0.5)

gray = num / den

cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

cv2.imwrite("./debug.png", gray * 255)


