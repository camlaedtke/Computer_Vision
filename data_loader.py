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


class ImageLoader(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_height, img_width, n_classes, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size, self.img_height, self.img_width, 3), dtype=np.float32)
        y = np.zeros((self.batch_size, self.img_height, self.img_width, n_classes), dtype='uint8')
        
        for j, path in enumerate(batch_input_img_paths):
            img = img_to_array(load_img(path))
            img = resize(img, (self.img_height, self.img_width, 3), mode='constant', preserve_range=True)
            img = img / 255
            x[j] = img
            
            
        for j, path in enumerate(batch_target_img_paths):
            mask = img_to_array(load_img(path, color_mode='grayscale'))
            mask = cv2.resize(mask, (self.img_height, self.img_width), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, 2)
            mask = to_categorical(mask, self.n_classes)
            y[j] = mask
      
        return np.array(x), np.array(y)


def sort_img_paths(input_dir, target_dir):
    # input_dir = "downloaded_datasets\\oxford_pets\\images\\"
    # target_dir = "downloaded_datasets\\oxford_pets\\annotations\\trimaps\\"

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    return input_img_paths, target_img_paths



def shuffle_img_paths(input_img_paths, target_img_paths):
    # Split our img paths into a training and a validation set
    val_samples = 1000
    randomseed = 1337
    random.Random(randomseed).shuffle(input_img_paths)
    random.Random(randomseed).shuffle(target_img_paths)

    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]

    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]
    
    return train_input_img_paths, train_target_img_paths, val_input_img_paths, val_target_img_paths
    

    
    
def get_pets_data(path, img_height, img_width, n_classes, subset=None):
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
        
    if (subset is not None):
        X = np.zeros((subset, img_height, img_width, 3), dtype=np.float32)
        y = np.zeros((subset, img_height, img_width, n_classes), dtype=np.float32)
        print("Number of images: " + str(subset))
    else:
        X = np.zeros((len(ids), img_height, img_width, 3), dtype=np.float32)
        y = np.zeros((len(ids), img_height, img_width, n_classes), dtype=np.float32)
        print("Number of images: " + str(len(ids)))
        
    
    print(y.shape)
        
    for n, id_ in enumerate(ids):
        print("\r Loading %s \ %s " % (n, len(ids)), end='')
        
        # load images
        img = load_img(path + "images\\" + id_)
        x_img = img_to_array(img)
        x_img = resize(x_img, (img_height, img_width, 3), mode='constant', preserve_range = True)
        
        # load masks
        id_mask = id_[:-4] + ".png"
        mask = img_to_array(load_img(path + "annotations\\trimaps\\" + id_mask, color_mode = "grayscale"))
        mask = cv2.resize(mask, (img_height, img_width), interpolation = cv2.INTER_NEAREST)
        # mask = mask - 1
        mask = np.expand_dims(mask, 2)
        mask = to_categorical(mask, n_classes)
        
        # save images
        X[n, ...] = x_img.squeeze()
        y[n] = mask.astype(np.uint8)
        
        if (subset is not None) and (n == subset-1):
            break
            
    print("Done!")
    return np.array(X), np.array(y)



def expand_mask_channels(y_train, y_test, n_classes):
    y_train_reshaped = np.zeros((TRAIN_LENGTH, img_width, img_height), dtype=np.float32)
    for idx, mask in enumerate(y_train):
        y_train_reshaped[idx] = cv2.resize(mask, (img_width, img_height)) # this should be method=internearest

    y_test_reshaped = np.zeros((TEST_LENGTH, img_width, img_height), dtype=np.float32)
    for idx, mask in enumerate(y_test):
        y_test_reshaped[idx] = cv2.resize(mask, (img_width, img_height))   

    train_masks = np.zeros((TRAIN_LENGTH, img_width, img_height, n_classes), dtype=np.float32)
    for idx, mask in enumerate(y_train):
        mask[mask < 1/255] = 0
        mask.astype(np.int)
        train_masks[idx] = to_categorical(mask, n_classes)

    test_masks = np.zeros((TEST_LENGTH, img_width, img_height, n_classes), dtype=np.float32)
    for idx, mask in enumerate(y_test):
        mask[mask < 1/255] = 0
        mask.astype(np.int)
        test_masks[idx] = to_categorical(mask, n_classes)
    return train_masks, test_masks



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






