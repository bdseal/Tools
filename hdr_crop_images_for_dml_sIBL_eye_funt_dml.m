close all;
%% batch read
% DML/;hdr_gallery/;Funt/;HDR_Eye/;stanford/;sIBL
%imgPath = '/media/sensetime/My Passport/git/hdr_training/sitmo_training_full_images/HDR/stanford/';
% imgPath = 'L:\git\hdr_training\sitmo_training_full_images\HDR\Funt\';
imgPath = 'L:\git\hdr_testing\hdr_eye_9_bracket\HDR\'
imgDir  = dir([imgPath '*.hdr']); 
num_test=2000;
count=0;
for num_of_series_i = 1:length(imgDir)          % 遍历结构体就可以�?��处理图片�?    
img = hdrread([imgPath imgDir(num_of_series_i).name]); %读取每张图片
%// loop over the image blocks
      
[rows,cols,k]= size(img);
block_height     = 100;
block_width      = 100;

blocks_per_row   =fix(rows/block_height);
blocks_per_col   = fix(cols/block_width);
number_of_blocks = blocks_per_row*blocks_per_col;

%// loop over the image blocks
for i = 1:blocks_per_row
    for j = 1:blocks_per_col
        %// get the cropped image from the original image
        idxI = 1+(i-1)*block_height;
        idxJ = 1+(j-1)*block_width;
        cropped_image_s= imcrop(img,[idxJ idxI block_height-1 block_width-1]);
 %// write the cropped image to the current folder
 %target and source image patches save to target/source respectively
 %---make test and train patches

if count<=num_test
            mkdir test_data ;directory_s=[cd,'/test_data/'];
            filename_s = sprintf('%d.hdr',count);
            hdrwrite(cropped_image_s,[directory_s,filename_s]);
            
else
             mkdir training_data ;directory_s=[cd,'/training_data/'];
            filename_s = sprintf('%d.hdr',count-num_test);
            hdrwrite(cropped_image_s,[directory_s,filename_s]);       
end

count = count + 1;
end
end%end of batch read
fid=fopen('log.txt', 'at+'); %打开文件
fprintf(fid, 'Pair_%d was successfully processed.\n',num_of_series_i);
fprintf(fid, 'The first number of next pair is %d .\n',count);
fclose(fid); 
end