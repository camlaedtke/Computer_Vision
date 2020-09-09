import os
import sys
import cv2
import PIL
import glob
import random
import imageio
import sklearn
import itertools
import numpy as np

from skimage.transform import resize
from skimage.morphology import label

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                              
    for subdir in subdirs:                                                                                     
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                       
                r.append(os.path.join(subdir, file))                                                                       
    return r 


def populate_directory(input_dir, annotation_dir, image_dir):
    
    ids_temp = list_files(input_path)
    
    mask_list = []
    for i in ids_temp:
        if i.endswith("labelIds.png"):
            mask_list.append(i)
            
    image_list = []
    for i in ids_temp:
        if i.endswith("leftImg8bit.png"):
            image_list.append(i)
            
                    
    for n, mask_id in enumerate(mask_list):
        img = load_img(mask_id, color_mode = "grayscale")
        print("\r saving {} / {}".format(n+1, len(mask_list)), end='')
        
        id_temp = mask_id.split("\\")
        id_temp = id_temp[-1]
        
        img.save(annotation_dir + id_temp)
        
        
    for n, img_id in enumerate(image_list):
        img = load_img(img_id, color_mode = "rgb")
        print("\r saving {} / {}".format(n+1, len(image_list)), end='')
        
        id_temp = img_id.split("\\")
        id_temp = id_temp[-1]
        
        img.save(image_dir + id_temp)
        
    print("\n done!")
    
    
    
def normalize_channels(X_train, X_test):
    
    R_MEAN = np.mean(X_train[:,:,:,0])
    G_MEAN = np.mean(X_train[:,:,:,1])
    B_MEAN = np.mean(X_train[:,:,:,2])
    
    print("Mean value of first channel: {}".format(R_MEAN))
    print("Mean value of second channel: {}".format(G_MEAN))
    print("Mean value of third channel: {}".format(B_MEAN))
    
    R_STD = np.std(X_train[:,:,:,0])
    G_STD = np.std(X_train[:,:,:,1])
    B_STD = np.std(X_train[:,:,:,2])
    
    print("Std of first channel: {}".format(R_STD))
    print("Std of second channel: {}".format(G_STD))
    print("Std of third channel: {}".format(B_STD))
    
    X_train[:,:,:,0] -= R_MEAN
    X_train[:,:,: 1] -= G_MEAN
    X_train[:,:,: 2] -= B_MEAN
    
    X_train[:,:,:,0] /= R_STD
    X_train[:,:,: 1] /= G_STD
    X_train[:,:,: 2] /= B_STD
    
    X_test[:,:,:,0] -= R_MEAN
    X_test[:,:,: 1] -= G_MEAN
    X_test[:,:,: 2] -= B_MEAN
    
    X_test[:,:,:,0] /= R_STD
    X_test[:,:,: 1] /= G_STD
    X_test[:,:,: 2] /= B_STD
    
    return X_train, X_test