##################################################
# Caution: returned LDRs are sorted by L-M-H, RGB
##################################################
from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2,sys,os,random,linecache,glob 
from scipy import misc,interpolate
from skimage.measure import compare_ssim as ssim

def getCRF(dorfCurves_FILE,Nth):
    I = linecache.getline(dorfCurves_FILE, 6*Nth+4) 
    B = linecache.getline(dorfCurves_FILE, 6*Nth+6)
    name_I = [float(x) for x in I.split( )]
    name_B = [float(x) for x in B.split( )]
#    return name_I.split( ),name_B.split( )
    return np.asarray(name_I),np.asarray(name_B)

def getFileName(path):

    f_list = os.listdir(path)
    s=[]
    for i in f_list:
        s.append(i)
    return s

def CRF(x):
    "Please modify this function according to your CRFs"
    "Gamma Correction can be used to fit one CRF curve."
    return x**2.2

def createLDRs(ref_HDR,out_name,dorfCurves_FILE,Nth):
    #refer to Paper:http://www.npal.cs.tsukuba.ac.jp/~endo/projects/DrTMO/
    #Thanks for help from Yuki Endo.
    R_=range(-5,3)

    taus = [np.math.sqrt(2.)**x for x in R_]

    img =ref_HDR
    img = 0.5 * img / img.mean()

    name_I,name_B=getCRF(dorfCurves_FILE,Nth)

    f = interpolate.interp1d(name_I,name_B, 'quadratic')

    h, w, c = ref_HDR.shape
    out_LDRs=np.zeros((h, w, c*len(R_)))

    for j, tau in enumerate(taus):
         # out = np.clip((255.*CRF(tau*img)),0., 255.).astype('uint8')      
         ynew=255.0*f(np.clip(tau*img, 0., 1.0))
         out_LDR = np.clip(ynew, 0., 255.0).astype(np.uint8)

         # print(out_LDR)
         output_path = "./out_LDRs/"+out_name
         if not os.path.exists(output_path):
            os.makedirs(output_path)
         # cv2.imwrite(output_path+'/output_'+ out_name +'_'+str(j)+'.tif', out_LDR)
         out_LDRs[:,:,j*c:(j+1)*c] = out_LDR# stack N LDR images along channel
    out_LDRs = out_LDRs.astype(np.uint8)  
    return out_LDRs

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
'''
Dynamic_Scenes_HDRs
Fairchild_HDR_format
HDR_Eye
sIBLing, hdr_gallery, MPII_HDR_format
'''
dataset='Dynamic_Scenes_HDRs'
# data_dir = '/home/xwh/Dataset/HDR/Train/'+dataset
# out_dir = '/home/xwh/SITMO/sitmo_DeepHDR/tf_records/256_64_tfrecords_'+dataset
data_dir = '/home/xwh/SITMO/sitmo_DeepHDR/dataset/Test/out_LDRs_Dynamic_Scenes_HDRs/'
out_dir = '/home/xwh/SITMO/sitmo_DeepHDR/dataset/tf_records_test/256_64_tfrecords_'+dataset+'_test'
dorfCurves_FILE='./dorfCurves.txt'
Nth=random.randint(0,200)#[0,200]

scene_dirs = [scene_dir for scene_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, scene_dir))]
scene_dirs = sorted(scene_dirs)
num_scenes = len(scene_dirs)

patch_size = 256
patch_stride = 64
batch_size = 20

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

count = 0
cur_writing_path = os.path.join(out_dir, "train_{:d}_{:04d}.tfrecords".format(patch_stride, 0))
writer = tf.python_io.TFRecordWriter(cur_writing_path)

for i, scene_dir in enumerate(scene_dirs):
    if (i%10 == 0):
        print('%d/%d' %(i, num_scenes))
    
    # read images
    # assume the exposure increases with file name
    cur_dir = os.path.join(data_dir, scene_dir)
    scene_dirs = [scene_dir for scene_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, scene_dir))]
    ref_HDR_path = sorted(glob.glob(os.path.join(cur_dir, '*.hdr')))
    for num_HDR_cur_dir in range (len(ref_HDR_path)):
        ref_HDR_name=ref_HDR_path[num_HDR_cur_dir]
        print(ref_HDR_name)
        ref_HDR = cv2.imread(ref_HDR_name, -1).astype(np.float32) # read raw values
        h, w, c = ref_HDR.shape
        in_LDRs = createLDRs(ref_HDR, ref_HDR_name.split('/')[-1].split('.')[0], dorfCurves_FILE, Nth)
   

        def write_example(h1, h2, w1, w2):
            global count
            global writer
            
            cur_batch_index = count // batch_size

            if count%batch_size == 0:
                writer.close()
                cur_writing_path = os.path.join(out_dir, "train_{:d}_{:04d}.tfrecords".format(patch_stride, cur_batch_index))
                writer = tf.python_io.TFRecordWriter(cur_writing_path)

            # reverting them from BGR to RGB
            in_LDRs_patch = in_LDRs[h1:h2, w1:w2, :]
            ref_HDR_patch = ref_HDR[h1:h2, w1:w2, ::-1]

            count += 1

            # create example
            example = tf.train.Example(features=tf.train.Features(feature={
                'in_LDRs':bytes_feature(in_LDRs_patch.tostring()),
                'ref_HDR': bytes_feature(ref_HDR_patch.tostring()),
                }))
            writer.write(example.SerializeToString())

        # generate patches
        for h_ in range(0, h-patch_size+1, patch_stride):
            for w_ in range(0, w-patch_size+1, patch_stride):
                write_example(h_, h_+patch_size, w_, w_+patch_size)

        # deal with border patch
        if h%patch_size:
            for w_ in range(0, w-patch_size+1, patch_stride):
                write_example(h-patch_size, h, w_, w_+patch_size)

        if w%patch_size:
            for h_ in range(0, h-patch_size+1, patch_stride):
                write_example(h_, h_+patch_size, w-patch_size, w)

        if w%patch_size and h%patch_size :
            write_example(h-patch_size, h, w-patch_size, w)

writer.close()
print("Finished!\nTotal number of patches:", count)
