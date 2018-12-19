from somefunctions import *
import scipy

###############################################################################
###########             PARAMETERS              ###############################
###############################################################################

# Model parameters
pool_size = (2, 2)
train_shape = 400 
patch_size = 16
input_size = 20
pad_size = int(input_size/2 - patch_size/2)
pad_rotate_size = int( input_size / np.sqrt(2) ) + 2
final_layer_units = 2


# Training parameters
reg = 1e-5 
learning_rate = 0.001
epochs = 20
batch_size = 250
steps_per_epoch = 2
validation_ratio = 0.8


# Data augmentation parameters
FLIP_FLAG = True # add random flips to the patches
ROTATION_90_FLAG = True # add rotation of 90 degrees to the patches
BRIGHT_CONTRAST_FLAG = True # modify randomly the brightness and the constrast


#Other stuff
NameWeights = 'model0_weights'
SubmissionName = 'model0_Submission.csv'
PredictionName = 'model0_prediction'

###############################################################################
###########              MODEL CREATION              ##########################
###############################################################################

def CreateModel():
    '''Create a sequential model'''        
    model = Sequential()
    
    model.add(Convolution2D(2, (3,3), 
                            input_shape = ( input_size, input_size, 3),
                            padding = 'SAME', activation = 'relu',
                            kernel_initializer = K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                           ))
    
    model.add(Flatten())     
    model.add(Dense(units = 2, activation = 'sigmoid'))

    # Choose optimizer and callbacks
    opt = Adam(lr=learning_rate) 
    lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5,
                                    verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    stop_callback = EarlyStopping(monitor='acc', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    
    # Compile
    model.compile(loss=binary_crossentropy,
                  optimizer=opt,
                  metrics=['acc'])
    
    return model, stop_callback, lr_callback