import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from keras.layers import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.losses import binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def pick_test_images():
    test_imgs = []
    for i in range(1, 51):
        name = '../Data/test_set_images/test_'+str(i)+'/test_' + str(i) + '.png'
        test_imgs.append(load_image(name))
    return test_imgs

def value_to_class(v):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

    
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def MY_mask_to_submission_strings(mask1D, fileNumber):
    img_number = fileNumber
    patch_size = 16
    test_size = 608
    im = label_to_img(test_size, test_size, patch_size, patch_size, mask1D)
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number+1, j, i, label))


def MY_masks_to_submission(submission_filename, masks1D):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for idx,fn in enumerate(masks1D):
            f.writelines('{}\n'.format(s) for s in MY_mask_to_submission_strings(fn,idx))
            
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def padding_imgs(imgs,pad_size):
    length_padded_image = imgs.shape[1] + 2*pad_size
    height_padded_image = imgs.shape[2] + 2*pad_size
    X = np.empty((imgs.shape[0],length_padded_image,height_padded_image,3))
    #pad the images
    for i in range(imgs.shape[0]):
            X[i] = cv2.copyMakeBorder(imgs[i],pad_size,pad_size,pad_size,pad_size,cv2.BORDER_REFLECT_101)
    return X

def padding_GT(imgs,pad_size):
    length_padded_image = imgs.shape[1] + 2*pad_size
    height_padded_image = imgs.shape[2] + 2*pad_size
    X = np.empty((imgs.shape[0],length_padded_image,height_padded_image))
    #pad the images
    for i in range(imgs.shape[0]):
            X[i] = cv2.copyMakeBorder(imgs[i],pad_size,pad_size,pad_size,pad_size,cv2.BORDER_REFLECT_101)
    return X