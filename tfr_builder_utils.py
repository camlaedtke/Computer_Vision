import os
import re
import sys
import cv2
import PIL
import json
import math
import time
import random
import sklearn
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.transform import resize

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img



def load_image_rgb_data(fp):
    # Opening JSON file 
    with open(fp, 'r') as openfile: 
        # Reading from json file 
        image_info = json.load(openfile) 
    info_dict = {
        "R_MEAN": float(image_info["R_MEAN"]),
        "G_MEAN": float(image_info["G_MEAN"]),
        "B_MEAN": float(image_info["B_MEAN"]),
        "R_STD": float(image_info["R_STD"]),
        "G_STD": float(image_info["B_STD"]),
        "B_STD": float(image_info["G_STD"]),
    }
    return info_dict


def normalize_image_channels(x_img, rgb_data):
    x_img[:,:,0] -= rgb_data['R_MEAN']
    x_img[:,: 1] -= rgb_data['G_MEAN']
    x_img[:,: 2] -= rgb_data['B_MEAN']

    x_img[:,:,0] /= rgb_data['R_STD']
    x_img[:,: 1] /= rgb_data['G_STD']
    x_img[:,: 2] /= rgb_data['B_STD']
    
    return x_img


def display(display_list):
    plt.figure(figsize=(15, 5))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def convertToNumber(s):
    return int.from_bytes(s.encode(), 'little')


def convertFromNumber(n):
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()


def extract_pets_data_info(path, subset=None):
    ids_temp = next(os.walk(path + "images"))[2]
    ids_1 = []
    for i in ids_temp:
        if i.endswith(".jpg"):
            ids_1.append(i)
            
    random.seed(2019)
    id_order = np.arange(len(ids_1))
    np.random.shuffle(id_order)
    
    ids = []
    for i in range(len(id_order)):
        ids.append(ids_1[np.int(id_order[i])])
        
    print("Number of images: " + str(len(ids)))
    
    image_data = []
            
    for n, id_ in enumerate(ids):
        print("\r Processing %s \ %s " % (n+1, len(ids)), end='')
        
        # load images
        img = load_img(path + "images\\" + id_)
        x_img = img_to_array(img)
        x_img = x_img.squeeze()
        
        # load masks
        id_mask = id_[:-4] + ".png"
        mask = img_to_array(load_img(path + "annotations\\trimaps\\" + id_mask, color_mode = "grayscale"))
        mask = mask.astype(np.uint8)
        
        # get size info
        img_height = x_img.shape[0]
        img_width = x_img.shape[1]
        img_depth = x_img.shape[2]
        mask_depth = mask.shape[2]
        
        # parse file info
        label = re.findall(r'\d+', id_)
        label = label[0]
        pos_label = id_.find(label)
        text = id_[0:pos_label]
        text = text[:-1]
        text_encoded = int.from_bytes(text.encode(), 'little') # convertToNumber(text)

        image_filename = path + "images\\" + id_
        mask_filename = path + "annotations\\trimaps\\" + id_mask
        
        # add to list of dicts
        image_dict = {
            "image_filename": image_filename,
            "mask_filename": mask_filename,
            "id": id_[:-4],
            "height": img_height,
            "width": img_width,
            "image_depth": img_depth,
            "mask_depth": mask_depth,
            "class_text_encoded": text_encoded,
            "class_label": int(label),
        }

        image_data.append(image_dict)
        
        if (subset is not None) and (n == subset-1):
            break
    
    return image_data