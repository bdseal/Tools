import numpy as np
import os
import glob
import shutil

#data_dir = 'L:\Datasets\Dynamic Scenes\Training'
data_dir = 'L:\Datasets\Dynamic Scenes\Test\PAPER'
ref_HDR = "HDRImg.hdr"
out_dir="L:\Datasets\Dynamic_Scenes_HDRs\\"
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

scene_dirs = [scene_dir for scene_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, scene_dir))]
scene_dirs = sorted(scene_dirs)
num_scenes = len(scene_dirs)

for i, scene_dir in enumerate(scene_dirs):
    print('%d/%d' %(i, num_scenes))
    
    # assume the exposure increases with file name
    cur_dir = os.path.join(data_dir, scene_dir)
    in_HDR = os.path.join(cur_dir, ref_HDR)
    #out_HDR_name='scene_%d_train'%i+'.hdr'
    out_HDR_name='scene_%d_test'%i+'.hdr'
    out_HDR = os.path.join(out_dir, out_HDR_name)
    shutil.copy(in_HDR, out_HDR)