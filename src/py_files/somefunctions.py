import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras import utils
from keras import backend as K
from keras.regularizers import l2
from keras.layers import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras import initializers as K_init

#########################################################################################
##################### GIVEN FUNCTIONS ###################################################
#########################################################################################

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

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

def value_to_class(v):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

    
def label_to_img(imgwidth, imgheight, w, h, labels):
    ''' From a vector of labels creates an image of (imgwidth x imgheight).
        Each label refers to a patch of dimension (w x h)
    '''
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

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

#########################################################################################
####################### OUR FUNCTIONS ###################################################
#########################################################################################

def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the split ratio. If ratio is 0.8 you will have 80% of your data 
    set dedicated to training and the rest dedicated to testing. 
    Return the training then testing sets (x_tr, x_te) and training then testing labels (y_tr, y_te).
    """
    #Set seed
    np.random.seed(seed)
    xrand = np.random.permutation(x)
    np.random.seed(seed)
    yrand = np.random.permutation(y)
    #Used to compute how many samples correspond to the desired ratio.
    limit = int(y.shape[0]*ratio)
    x_tr = xrand[:limit]
    x_te = xrand[(limit+1):]
    y_tr = yrand[:limit]
    y_te = yrand[(limit+1):]
    return x_tr, x_te, y_tr, y_te


def MY_masks_to_submission(submission_filename, masks1D):
    """ Converts the matrix containing all the labels into a submission file.
        
        masks1D should be a 2D array, where the ith row contains the labels of the ith test image.
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for idx,fn in enumerate(masks1D):
            f.writelines('{}\n'.format(s) for s in MY_mask_to_submission_strings(fn,idx))
      
    
def MY_mask_to_submission_strings(mask1D, fileNumber):
    '''To be called inside MY_masks_to_submission.
    
        Write the informations of the image fileNumber, starting from the array of labels.
    '''
    img_number = fileNumber
    patch_size = 16
    test_size = 608
    im = label_to_img(test_size, test_size, patch_size, patch_size, mask1D)
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number+1, j, i, label))

            
def pick_test_images(root = '../Data'):
    ''' Pick the images for the submission 
    '''
    test_imgs = []
    for i in range(1, 51):
        name = root + '/test_set_images/test_'+str(i)+'/test_' + str(i) + '.png'
        test_imgs.append(load_image(name))
    return test_imgs
    
    
def padding_imgs(imgs,pad_size):
    ''' Pad an array of RGB images using NumPy
    '''
    length_padded_image = imgs.shape[1] + 2*pad_size
    height_padded_image = imgs.shape[2] + 2*pad_size
    X = np.empty((imgs.shape[0],length_padded_image,height_padded_image,3))
    #pad the images
    for i in range(imgs.shape[0]):
        img = imgs[i]
        temp = np.empty( (length_padded_image, height_padded_image,3) )
        for j in range(3): #RGB channels
            temp[:,:,j] = np.pad(img[:,:,j],pad_size,'reflect')
        X[i] = temp
    return X


def padding_GT(imgs,pad_size):
    ''' Pad an array of groundtruth images using NumPy
    '''
    length_padded_image = imgs.shape[1] + 2*pad_size
    height_padded_image = imgs.shape[2] + 2*pad_size
    X = np.empty((imgs.shape[0],length_padded_image,height_padded_image))
    #pad the images
    for i in range(imgs.shape[0]):
            X[i] = np.pad(imgs[i],pad_size,'reflect')
    return X


def imgs_to_patches(imgs, img_size, patch_size, input_size):
    ''' Takes an array of images and outputs an array with the patches of (input_size x input_size) centered around the patches of 16x16
    '''
    pad_size = int(input_size/2 - patch_size/2)
    patches = []
    for idx in range(imgs.shape[0]):
        im = imgs[idx]
        for i in range(0,img_size,patch_size):
            for j in range(0,img_size,patch_size):
                temp = im[j:j+patch_size, i:i+patch_size, :]
                temp = temp.reshape(1,patch_size,patch_size,3)
                temp = padding_imgs(temp,pad_size)
                temp = temp.reshape(input_size,input_size,3)
                patches.append(temp)
    return np.asarray(patches)



def imgs_to_windows(imgs, img_size, patch_size, window_size):
    ''' Takes an array of padded images and outputs an array with the windows of (window_size x window_size) centered around the patches
    '''
    windows = []
    for idx in range(imgs.shape[0]):
        im = imgs[idx]
        for i in range(img_size//patch_size):
            for j in range(img_size//patch_size):
                temp = im[j*patch_size:window_size + j*patch_size,
                                  i*patch_size:window_size + i*patch_size]
                windows.append(temp)
    return np.asarray(windows)


def data_augmentation(X, rot_flag, flip_flag, bright_flag):
    '''Data augmentation on X, element of size (input_size * input_size * 3)'''
    #flip
    if flip_flag:
        flip_decision = np.random.choice(3)
        if flip_decision == 1:
            X = np.flipud(X)
        if flip_decision == 2: 
            X = np.fliplr(X)
    
    #rotate
    if rot_flag:
        number_of_rotations = np.random.choice(3)
        X = np.rot90(X, number_of_rotations)
    
    #contrast and brightness
    if bright_flag:
        brightness = np.random.rand()*0.3 - 0.15
        contrast = np.random.rand()*0.25 - 0.125
        X = np.clip( X * (contrast/0.5+1) - contrast + brightness, 0, 1)
        
    return X

def crop_center(img,cropx,cropy):
    if len(img.shape) == 3:
        y,x, _ = img.shape
    if len(img.shape) == 2:
        y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]    