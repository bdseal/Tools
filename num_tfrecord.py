import tensorflow as tf
import os
path = "/home/xwh/SITMO/sitmo_DeepHDR/dataset/tf_records/256_64_tfrecords/" 
files= os.listdir(path) 
n = 0
for file in files:
    if not os.path.isdir(file): 
         tf_records_filenames=path+"/"+file; 
         c = 0
         for record in tf.python_io.tf_record_iterator(tf_records_filenames):
         	print (record)
          	c += 1 
         # print c
    n+=c
print(n) 