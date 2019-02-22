'''
For under-saturation images, intensity less than 5 is assigned as true, otherwise false;
For over-saturation images, intensity higher than 225 is assigned as true, otherwise false.
'''

import cv2
import os
import numpy as np

flag='under'#or 'under'
rootdir = './%s-exposures/'%(flag)
outdir = './%s-mask/'%(flag)

if not os.path.exists(outdir):
    os.makedirs(outdir)

list = os.listdir(rootdir)
list.sort()
print(list)
for i in range(0,len(list)):
	path = os.path.join(rootdir,list[i])
	if os.path.isfile(path):
		im=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		if flag=='over':
			globals()["%s_flag"%(flag)]=im>225
		else:
			globals()["%s_flag"%(flag)]=im<5
		print(outdir+'%i.png'%i)
		out=(globals()["%s_flag"%(flag)]+0)*255
		cv2.imwrite(outdir+'%i.png'%i, out)



# from PIL import Image
# img = Image.open('test.jpg')
 
# L-Gray
# Img = img.convert('L')
# Img.save("test1.jpg")
 

# threshold = 225
 
# table = []
# for i in range(256):
#     if i < threshold:
#         table.append(0)
#     else:
#         table.append(1)
 
# photo = Img.point(table, '1')
# photo.save("test2.jpg")