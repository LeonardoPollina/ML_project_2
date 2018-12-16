from utilities import *
from f1_score import *


###############################################################################
###########             PARAMETERS              ###############################
###############################################################################

# Model parameters
pool_size = (2, 2)
train_shape = 400 
patch_size = 16
input_size = 80
pad_size = int(input_size/2 - patch_size/2)
pad_rotate_size = int( input_size / np.sqrt(2) ) + 2
final_layer_units = 2


# Training parameters
reg = 1e-5  
learning_rate = 0.005
epochs = 200
batch_size = 128
steps_per_epoch = 50


# Data augmentation parameters
FLIP_FLAG = True # add random flips to the patches
BRIGHT_CONTRAST_FLAG = True # modify randomly the brightness and the constrast


#Other stuff
NameWeights = 'model_1512_Weights'
SubmissionName = 'model_1512_Submission.csv'
PredictionName = 'model_1512_prediction'



###############################################################################
###########             DATA GENERATION              ##########################
###############################################################################

def MinibatchGenerator(X,Y):
    '''
    MinibatchGenerator generates a minibatch starting from the padded images X and
    the padded groundtruth images Y.
    Since it was impossible to store all the training data, we generate them
    dynamically with this function.
    '''
    while 1:        
        # Initialize the minibatch
        X_batch = np.empty((batch_size, input_size, input_size, 3))
        Y_batch = np.empty((batch_size,2))

        # We want to select randomly the patches inside the images. So we will
        # select a random pixel in the image and crop around it a square of the
        # correct size. To be able to crop we select pixels in between low and high
        low=pad_rotate_size + patch_size // 2
        high = pad_rotate_size + train_shape - patch_size // 2

        for i in range(batch_size):
            # Select a random image
            idx = np.random.choice(X.shape[0])
            # Select a random pixel
            x_coord = np.random.randint(low=low, high = high ) 
            y_coord = np.random.randint(low=low, high = high )
            # Crop around the random pixel (imgs)
            X_temp = X[idx,x_coord - pad_rotate_size:x_coord + pad_rotate_size,
                           y_coord - pad_rotate_size:y_coord + pad_rotate_size]
            # Arbitrary rotation and correct crop of X_temp
            degree = np.random.choice(180)
            X_temp = scipy.ndimage.interpolation.rotate(X_temp, degree)
            X_temp = crop_center(X_temp,input_size,input_size)
            # Data augmentation
            X_temp = data_augmentation(X_temp, False, FLIP_FLAG, BRIGHT_CONTRAST_FLAG)
            X_batch[i] = X_temp
            # Crop around the random pixel (gt_imgs)
            gt_temp = Y[idx,x_coord - pad_rotate_size:x_coord + pad_rotate_size,
                            y_coord - pad_rotate_size:y_coord + pad_rotate_size] 
            # Arbitrary rotation and correct crop of gt_temp
            # Same degree
            gt_temp = scipy.ndimage.interpolation.rotate(gt_temp, degree)
            gt_temp = crop_center(gt_temp,patch_size,patch_size)
            Y_batch[i] = utils.to_categorical(patch_to_label(gt_temp),2)
            
        yield X_batch, Y_batch

        

###############################################################################
###########              MODEL CREATION              ##########################
###############################################################################

def CreateModel():     
    model = Sequential()
    
    model.add(Convolution2D(64, (5,5), 
                            input_shape = ( input_size, input_size, 3),
                            padding = 'SAME', activation = 'relu',
                            kernel_initializer = K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                           ))  
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5)) 

    model.add(Convolution2D(128, (3,3), 
                            padding = 'SAME', activation = 'relu',
                            kernel_initializer = K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                           ))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(256, (3,3),
                            padding = 'SAME', activation = 'relu',
                            kernel_initializer = K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                           ))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5)) 

    model.add(Convolution2D(256, (3,3),
                            padding = 'SAME', activation = 'relu',
                            kernel_initializer = K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                           ))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5)) 

    model.add(Flatten())       
    model.add(Dense(128, activation = 'relu', kernel_regularizer = l2(reg)))
    model.add(Dropout(0.5))       
    model.add(Dense(units = 2, activation = 'softmax', kernel_regularizer = l2(reg)))

    # Optimizer and callbacks        
    opt = Adam(lr=learning_rate) # Adam optimizer with default initial learning rate
    lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=15,
                                    verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    stop_callback = EarlyStopping(monitor='acc', min_delta=0.0001, patience=20, verbose=1, mode='auto')
    
    model.compile(loss=categorical_crossentropy,
                  optimizer=opt,
                  metrics=['acc'])
    
    return model, stop_callback, lr_callback