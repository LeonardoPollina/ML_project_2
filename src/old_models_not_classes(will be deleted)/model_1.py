# Functions to generate the model with rotations of 90 degrees

from somefunctions import *
from f1_score import *

###############################################################################
###########             PARAMETERS              ###############################
###############################################################################

# Model parameters
pool_size = (2, 2)
train_shape = 400 #size of the training images
patch_size = 16
input_size = 64
pad_size = int(input_size/2 - patch_size/2)


# Training parameters
reg = 1e-5  #regularization term
learning_rate = 0.001
epochs = 45 #very small, only preliminary tests
batch_size = 256
steps_per_epoch = 125 #the number of training samples is huge, arbitrary value


# Data augmentation parameters
FLIP_FLAG = True # add random flips to the patches
ROTATION_FLAG = True # add random rotation to the patches
BRIGHT_CONTRAST_FLAG = True # modify randomly the brightness and the constrast


#Other stuff
NameWeights = 'NicolaWeights'
SubmissionName = 'NicolaSubmission.csv'



###############################################################################
###########             DATA GENERATION              ##########################
###############################################################################




def generate_minibatch(X,Y):
    """
    Generate a minibatch
    """
    while 1:
        # Generate one minibatch
        X_batch = np.empty((batch_size, input_size, input_size, 3))
        Y_batch = np.empty(batch_size)
        low=input_size//2
        high = (train_shape + 2*pad_size - input_size//2)
        for i in range(batch_size):
            # Select a random image
            idx = np.random.choice(X.shape[0])
            
            x_coord = np.random.randint(low=low, high = high ) 
            y_coord = np.random.randint(low=low, high = high )
      
            X_temp = X[idx,x_coord - input_size//2:x_coord + input_size//2,
                           y_coord - input_size//2:y_coord + input_size//2]
            X_batch[i] = data_augmentation(X_temp, ROTATION_FLAG, FLIP_FLAG, BRIGHT_CONTRAST_FLAG)
            
            gt_temp = Y[idx,x_coord - patch_size//2:x_coord + patch_size//2,
                            y_coord - patch_size//2:y_coord + patch_size//2]            
            Y_batch[i] = patch_to_label(gt_temp)
            
        yield X_batch, Y_batch


###############################################################################
###########              MODEL CREATION              ##########################
###############################################################################

def CreateModel():
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
    
    
    model, stop_callback, lr_callback = CreateModel()
    
    np.random.seed(20122018) # Reproducibility + remember the deadline is the 20.12.2018
    
    try:
        model.fit_generator(generate_minibatch(X,Y),
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[lr_callback, stop_callback])
    except KeyboardInterrupt:
        print('\n\nKeyboard interruption!\n\n')
        pass
    

    model.save_weights(NameWeights)
    
    print(f'Training completed, weights saved in: {NameWeights}')
    
    return model