# import the necessary packages
import cv2,os
import numpy as np

g = os.walk(r"./HDR/sIBL/") 
out_path='./training_data/'
count=70058
for path,dir_list,file_list in g:  
    for file_name in file_list:  
        print(os.path.join(path, file_name) )
        # load the image and show it
        filesuffix = os.path.splitext(file_name)[1][1:]
        if  filesuffix in ['hdr']: 
            image = cv2.imread(os.path.join(path, file_name),-1)       
            
            [rows,cols,k]= np.shape(image);
            block_height     = 100#512;
            block_width      = 100#512;
            
            blocks_per_row   =int(np.fix(rows/block_height));
            blocks_per_col   = int(np.fix(cols/block_width));
            number_of_blocks = blocks_per_row*blocks_per_col;
            
            # loop over the image blocks
            for i in range (1,blocks_per_row+1):
                for j in range (1,blocks_per_col+1):
                    idxI = (i-1)*block_height;#python start from 0
                    idxJ = (j-1)*block_width;
                    cropped= image[idxJ:idxJ+block_height, idxI:idxI+block_width];
                    cv2.imwrite(out_path+str(count)+".hdr", cropped);
                    print (np.shape(cropped))
                    count=count+1;
#cv2.imshow("cropped", cropped)
#cv2.waitKey(0)

# write the cropped image to disk in PNG format
