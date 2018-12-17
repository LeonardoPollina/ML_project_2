import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from sklearn.metrics import f1_score as f1_score_sklearn
import scipy

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
from keras.layers import Input, concatenate
from keras.models import Model
from keras.layers import BatchNormalization

################################################################################
##################### GIVEN FUNCTIONS ##########################################
################################################################################

# percentage of pixels > 1 required to assign a foreground label to a patch
foreground_threshold = 0.25 

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

################################################################################
####################### OUR FUNCTIONS ##########################################
################################################################################
def get_idx_split_data(N, ratio, seed=1):
    """
    Split the dataset based on the split ratio. N is the number of data that 
    we want to split.
    Return random indices to select the train/validation set.
    """
    #Set seed
    np.random.seed(seed)
    idx_permuted = np.random.permutation(np.arange(N))
    #Used to compute how many samples correspond to the desired ratio.
    limit = int(N*ratio)
    idx_tr = idx_permuted[:limit]
    idx_te = idx_permuted[(limit+1):]
    return idx_tr, idx_te
    
def MY_masks_to_submission(submission_filename, masks1D):
    ''' Converts the matrix containing all the labels into a submission file.
    This function relies on the function MY_mask_to_submission_strings
  
    NOTE: masks1D should be a 2D array, where the ith row contains the labels of 
    the ith test image.
    '''
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for idx,fn in enumerate(masks1D):
            f.writelines('{}\n'.format(s) for s in MY_mask_to_submission_strings(fn,idx))
      
    
def MY_mask_to_submission_strings(mask1D, img_number):
    '''To be called inside MY_masks_to_submission.
    Write the informations of the image img_number, starting from the 1D array 
    containing the predicted labels of the image.
    '''
    patch_size = 16
    test_size = 608
    im = label_to_img(test_size, test_size, patch_size, patch_size, mask1D)
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number+1, j, i, label))

            
def pick_test_images(root = '../Data'):
    ''' Pick the images, located in the root directory. 
    '''
    test_imgs = []
    for i in range(1, 51):
        name = root + '/test_set_images/test_'+str(i)+'/test_' + str(i) + '.png'
        test_imgs.append(load_image(name))
    return np.asarray(test_imgs)
    
    
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
    ''' Pad an array of 1 channel images using NumPy
    '''
    length_padded_image = imgs.shape[1] + 2*pad_size
    height_padded_image = imgs.shape[2] + 2*pad_size
    X = np.empty((imgs.shape[0],length_padded_image,height_padded_image))
    #pad the images
    for i in range(imgs.shape[0]):
            X[i] = np.pad(imgs[i],pad_size,'reflect')
    return X


#I THINK THAT WE DON'T USE THIS FUNCTION
# def imgs_to_patches(imgs, img_size, patch_size, input_size):
#     ''' Takes an array of images (both 1 or 3 channels) and outputs an array 
#     with the patches of (input_size x input_size) centered around the patches of 16x16
#     '''
#     pad_size = int(input_size/2 - patch_size/2)
#     patches = []
#     for idx in range(imgs.shape[0]):
#         im = imgs[idx]
#         for i in range(0,img_size,patch_size):
#             for j in range(0,img_size,patch_size):
#                 temp = im[j:j+patch_size, i:i+patch_size, :]
#                 temp = temp.reshape(1,patch_size,patch_size,3)
#                 temp = padding_imgs(temp,pad_size)
#                 temp = temp.reshape(input_size,input_size,3)
#                 patches.append(temp)
#     return np.asarray(patches)



def imgs_to_inputs(imgs, img_size, patch_size, input_size):
    ''' Takes an array of properly padded images and outputs an array with
    patches of dimensions (input_size x input_size). These patches will be
    overlapped, but their central portions of size (patch_size x patch_size) 
    will not.

    Basically, instead of extracting the patches from the images and then pad 
    them, this function allows to extract patches of the desired dimension,
    centered around the non overlapped patches of dimensions 
    (patch_size x patch_size) that would fit in the original image.
    '''
    inputs = []
    # For each padded image
    for idx in range(imgs.shape[0]):
        im = imgs[idx]

        # Crop patches of (input_size x input_size), centered around the patches
        # of (patch_size x patch_size) that would fit in the original image
        for i in range(img_size//patch_size):
            for j in range(img_size//patch_size):
                temp = im[j*patch_size:input_size + j*patch_size,
                          i*patch_size:input_size + i*patch_size]
                inputs.append(temp)

    return np.asarray(inputs)


def data_augmentation(X, rot_flag, flip_flag, bright_flag, 
                      bright_range = 0.3, contr_range = 0.25):
    '''Data augmentation on X, element of size (input_size * input_size * 3).

    rot_flag: randomly add 0, 1, 2 or 3 rotations of 90° to X
    flip_flag: randomly decide to (or not to) flip X vertically or horizontally.
    bright_flag: randomly change constrast and brightness of the input X. The
        default ranges of contrast and brightness are decided based on several
        training results and on heuristic considerations. If the parameters are 
        too high, the images becomes not recognizable.
    '''
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
        brightness = np.random.rand()*bright_range - (bright_range/2)
        contrast = np.random.rand()*contr_range - (contr_range/2)
        X = np.clip( X * (contrast/0.5+1) - contrast + brightness, 0, 1)
        
    return X

def crop_center(img, cropx, cropy):
    ''' Crop a patch from img. Centered in the middle of the image and of 
    dimensions (cropx x cropy). Works both for 1 and 3 channel images.

    This function is used after a rotation of an arbitrary degree.
    '''
    if len(img.shape) == 3:
        y,x, _ = img.shape
    if len(img.shape) == 2:
        y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]    


def LoadImages(pad_size = 0, root_dir = "../Data/", verbose = 1):
    ''' Load images and pad them using mirror boundary conditions. If pad_size
    is zero, then the images are not padded.
    '''
    # Load images
    image_dir = root_dir + "/training/images/"
    files = os.listdir(image_dir)
    n = len(files)
    if verbose : print("Loading " + str(n) + " images")
    imgs = [load_image(image_dir + files[i]) for i in range(n)]
    # Load groundtruth
    gt_dir = root_dir + "/training/groundtruth/"
    if verbose : print("Loading " + str(n) + " groundtruth images")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
    # Padding
    if verbose : print('Padding images using pad of: ', pad_size)
    imgs = padding_imgs(np.array(imgs),pad_size)
    gt_imgs = padding_GT(np.array(gt_imgs),pad_size)
    # Print some infos
    if verbose : print('Shape of imgs: ',imgs.shape)
    if verbose : print('Shape of gt_imgs: ',imgs.shape)

    return imgs, gt_imgs

def VisualizePrediction(PredictionName, IDX, img_size, patch_size = 16, PLOT = True):
    ''' Load the predicion of one model, saved in a (pickle) file called
    PredictionName and return the IDXth predicted image.

    If PLOT is true we generate also a plot of the prediction.
    '''
    # Getting back the prediction:
    with open(PredictionName, 'rb') as f: 
        predicted_labels_1D = pickle.load(f)
    predicted_labels = predicted_labels_1D.reshape(50,-1)

    # Get the labels related to image IDX and create an image
    im = label_to_img(img_size, img_size, patch_size, patch_size, predicted_labels[IDX])

    if PLOT: plt.imshow(im)
    return im