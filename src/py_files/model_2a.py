# Model with arbitrary rotations, with one node at the end.

from somefunctions import *
from f1_score import *
import scipy

###############################################################################
###########             PARAMETERS              ###############################
###############################################################################

# Model parameters
pool_size = (2, 2)
train_shape = 400 #size of the training images
patch_size = 16
input_size = 64
pad_rotate_size = int( input_size / np.sqrt(2) ) + 2


# Training parameters
reg = 1e-5  #regularization term
learning_rate = 0.001
nb_epoch = 45 
batch_size = 250
steps_per_epoch = 125 #the number of training samples is huge, arbitrary value


# Data augmentation parameters
FLIP_FLAG = True # add random flips to the patches
BRIGHT_CONTRAST_FLAG = True # modify randomly the brightness and the constrast


#Other stuff
NameWeights = 'NicolaWeights'
SubmissionName = 'NicolaSubmission.csv'



###############################################################################
###########             DATA GENERATION              ##########################
###############################################################################

def generate_minibatch_with_arbitrary_rotation(X,Y):
    """
    Generate a minibatch
    """
    while 1:        
        # Generate one minibatch
        X_batch = np.empty((batch_size, input_size, input_size, 3))
        Y_batch = np.empty(batch_size)
        low=pad_rotate_size + patch_size // 2
        high = pad_rotate_size + train_shape - patch_size // 2
        for i in range(batch_size):
            # Select a random image
            idx = np.random.choice(X.shape[0])
            
            x_coord = np.random.randint(low=low, high = high ) 
            y_coord = np.random.randint(low=low, high = high )
            
            X_temp = X[idx,x_coord - pad_rotate_size:x_coord + pad_rotate_size,
                           y_coord - pad_rotate_size:y_coord + pad_rotate_size]
            
            #arbitrary rotation and crop of X_temp
            degree = np.random.choice(180)
            X_temp = scipy.ndimage.interpolation.rotate(X_temp, degree)
            X_temp = crop_center(X_temp,input_size,input_size)
            
            X_temp = data_augmentation(X_temp, False, FLIP_FLAG, BRIGHT_CONTRAST_FLAG)
            
            X_batch[i] = X_temp
            
            gt_temp = Y[idx,x_coord - pad_rotate_size:x_coord + pad_rotate_size,
                            y_coord - pad_rotate_size:y_coord + pad_rotate_size]  #TODO: reduce this crop size
            
            #arbitrary rotation and crop of gt_temp
            #same degree
            gt_temp = scipy.ndimage.interpolation.rotate(gt_temp, degree)
            gt_temp = crop_center(gt_temp,patch_size,patch_size)
            Y_batch[i] = patch_to_label(gt_temp)
            
        yield X_batch, Y_batch
        

###############################################################################
###########              MODEL CREATION              ##########################
###############################################################################

def create_model():
    '''Create a sequential model'''        
    model = Sequential()
    
    model.add(Convolution2D(64, (5,5), 
                            input_shape = ( input_size, input_size, 3),
                            padding = 'SAME', activation = 'relu',
                            kernel_initializer = K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                           ))
    
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Convolution2D(64, (3,3), 
                            input_shape = ( input_size, input_size, 3),
                            padding = 'SAME', activation = 'relu',
                            kernel_initializer = K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                           ))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Convolution2D(128, (3,3),
                            padding = 'SAME', activation = 'relu',
                            kernel_initializer = K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                           ))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Convolution2D(256, (3,3),
                            padding = 'SAME', activation = 'relu',
                            kernel_initializer = K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                           ))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu', kernel_regularizer = l2(reg)))
    model.add(Dropout(0.5))         
    model.add(Dense(256, activation = 'relu', kernel_regularizer = l2(reg)))
    model.add(Dropout(0.5))       
    model.add(Dense(units = 1, activation = 'sigmoid'))

    #Optimizer          
    opt = Adam(lr=learning_rate) # Adam optimizer with default initial learning rate
 

    # This callback reduces the learning rate when the training accuracy does not improve any more
    lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5,
                                    verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    
    # Stops the training process upon convergence
    stop_callback = EarlyStopping(monitor='acc', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    
    model.compile(loss=binary_crossentropy,
                  optimizer=opt,
                  metrics=['acc', f1_score])
    
    return model, stop_callback, lr_callback


###############################################################################
###########                  TRAINING                ##########################
###############################################################################    


def train(X, Y):    
    '''
    Generate an instance of the model an train the model on X, Y
    '''
    print('Training set shape: ', X.shape) 
    print(f'Batch_size: {batch_size} \nSteps per epoch: {steps_per_epoch} \n')
    
    
    model, stop_callback, lr_callback = create_model()
    
    np.random.seed(20122018) # Reproducibility + remember the deadline is the 20.12.2018
    
    try:
        model.fit_generator(generate_minibatch_with_arbitrary_rotation(X,Y),
                            steps_per_epoch=steps_per_epoch,
                            nb_epoch=nb_epoch,
                            verbose=1,
                            callbacks=[lr_callback, stop_callback])
    except KeyboardInterrupt:
        print('\n\nKeyboard interruption!\n\n')
        pass
    

    model.save_weights(NameWeights)
    
    print(f'Training completed, weights saved in: {NameWeights}')
    
    return model