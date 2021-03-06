# Model to train a RNN. 
#
# With this model (without post-processing) we obtained a CrowdAI F1-score 
# of: 0.806
#
# This class defines the parameter of the model. Moreover, 3 functions must be 
# defined in this class:
# 1. MinibatchGenerator: generate a minibatch starting from X and Y, the 
#    padded version of the images and groundtruth images
# 2. CreateModel: generate a Keras model
# 3. Summary: print some informations about the model


from utilities import *
from f1_score import *

class MODEL_CLASS:

    # INITIALIZATION: set some parameters of the model
    def __init__(self):
        # Short description
        self.WhoAmI = 'VGG-like model with skip connections (RNN).'

        # Model parameters
        self.pool_size = (2, 2)
        self.train_shape = 400 
        self.patch_size = 16
        self.input_size = 72
        self.pad_size = int(self.input_size/2 - self.patch_size/2)
        self.final_layer_units = 2

        # Training parameters
        self.reg = 1e-5
        self.learning_rate = 0.001
        self.epochs = 40
        self.batch_size = 128
        self.steps_per_epoch = 250


        # Data augmentation parameters
        self.FLIP_FLAG = True # add random flips to the patches
        self.BRIGHT_CONTRAST_FLAG = True # modify randomly the brightness 
                                         # and the constrast
        # If there will be arbitrary rotations, compute the pad_rotate_size
        # accordingly.  
        # NOTE: this padding allows us to crop portions of images wide enough to
        #       be rotated of any degree and then to crop from them a portion of
        #       the desired size, which is (input_size x input_size) 
        self.ARB_ROT_FLAG = True
        if self.ARB_ROT_FLAG:
            self.pad_rotate_size = int( self.input_size / np.sqrt(2) ) + 2
        else:
            self.pad_rotate_size = self.pad_size

        #Other stuff
        self.NameWeights = 'model_RNN_Weights'
        self.SubmissionName = 'model_RNN_Submission.csv'
        self.PredictionName = 'model_RNN_prediction'




    ############################################################################
    ###########             DATA GENERATION              #######################
    ############################################################################
    def MinibatchGenerator(self,X,Y):
        '''
        MinibatchGenerator generates a minibatch starting from the padded images
        X and the padded groundtruth images Y.
        Since it was impossible to store all the training data, we generate them
        dynamically with this function.
        '''
        while 1:        
            # Initialize the minibatch
            X_batch = np.empty((self.batch_size, self.input_size, 
                                self.input_size, 3))
            Y_batch = np.empty((self.batch_size,2))

            # We want to select randomly the patches inside the images. So we 
            # will select a random pixel in the image and crop around it a 
            # square of the correct size. To be able to crop we select pixels in 
            # between low and high
            low=self.pad_rotate_size + self.patch_size//2
            high = self.pad_rotate_size + self.train_shape - self.patch_size//2

            for i in range(self.batch_size):
                # Select a random image
                idx = np.random.choice(X.shape[0])
                # Select a random pixel
                x_coord = np.random.randint(low=low, high = high ) 
                y_coord = np.random.randint(low=low, high = high )
                # Crop around the random pixel (imgs)
                X_temp = X[idx,
                  x_coord - self.pad_rotate_size:x_coord + self.pad_rotate_size,
                  y_coord - self.pad_rotate_size:y_coord + self.pad_rotate_size]
                # Arbitrary rotation and correct crop of X_temp
                degree = np.random.choice(180)
                X_temp = scipy.ndimage.interpolation.rotate(X_temp, degree)
                X_temp = crop_center(X_temp,self.input_size,self.input_size)
                # Data augmentation
                X_temp = data_augmentation(X_temp, False, self.FLIP_FLAG, 
                                                      self.BRIGHT_CONTRAST_FLAG)
                X_batch[i] = X_temp
                # Crop around the random pixel (gt_imgs)
                gt_temp = Y[idx,
                  x_coord - self.pad_rotate_size:x_coord + self.pad_rotate_size,
                  y_coord - self.pad_rotate_size:y_coord + self.pad_rotate_size] 
                # Arbitrary rotation and correct crop of gt_temp
                # Same degree
                gt_temp = scipy.ndimage.interpolation.rotate(gt_temp, degree)
                gt_temp = crop_center(gt_temp,self.patch_size,self.patch_size)
                Y_batch[i] = utils.to_categorical(patch_to_label(gt_temp),2)
                
            yield X_batch, Y_batch

            

    ############################################################################
    ###########              MODEL CREATION              #######################
    ############################################################################
    def CreateModel(self):  
        ########################################################################   
        ########## root (X0): conv, pooling, conv, pooling  ####################
        inputs = Input(shape=(self.input_size, self.input_size, 3))
        x0 = Convolution2D(32, (5,5), 
                        input_shape = ( self.input_size, self.input_size, 3),
                        padding = 'SAME', activation = 'relu',
                        kernel_initializer = 
                        K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                        )                              (inputs)
        x0 = Convolution2D(32, (5,5),
                        padding = 'SAME', activation = 'relu',
                        kernel_initializer = 
                        K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                        )                              (x0)
        x0 = MaxPooling2D(pool_size = self.pool_size)  (x0)
        x0 = Dropout(0.5)                              (x0)

        x0 = Convolution2D(64, (3,3),
                        padding = 'SAME', activation = 'relu',
                        kernel_initializer = 
                        K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                        )                              (x0)
        x0 = MaxPooling2D(pool_size = self.pool_size)  (x0)
        x0 = Dropout(0.5)                              (x0)

        ########################################################################
        ########### left (xl) : conv, pooling, conv, pooling ###################
        xl = Convolution2D(128, (3,3), 
                        padding = 'SAME', activation = 'relu',
                        kernel_initializer = 
                        K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                        )                              (x0)
        xl = MaxPooling2D(pool_size = self.pool_size)  (xl)
        xl = Dropout(0.5)                              (xl)

        xl = Convolution2D(256, (3,3),
                        padding = 'SAME', activation = 'relu',
                        kernel_initializer = 
                        K_init.RandomUniform(minval=-0.05, maxval=0.05, seed=1)
                        )                              (xl)
        xl = MaxPooling2D(pool_size = self.pool_size)  (xl)
        xl = Flatten()                                 (xl)

        ########################################################################
        ########### right (xr): do nothing  ####################################
        xr = Flatten() (x0)

        ########################################################################
        ########### merge left and right  ######################################
        concat = concatenate([xl, xr])                                         
        concat = Dense(256, activation = 'relu', 
                       kernel_regularizer = l2(self.reg)) (concat)
        concat = Dropout(0.5)                             (concat)
        concat = Dense(units = 2, activation = 'softmax') (concat)

        ########################################################################
        ########## create the model ############################################
        model = Model(inputs=inputs, outputs=concat)


        # Optimizer and callbacks        
        opt = Adam(lr=self.learning_rate)
        lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=15,
                                        verbose=1, mode='auto', 
                                        min_delta=0.0001, cooldown=0, min_lr=0)
        stop_callback = EarlyStopping(monitor='acc', min_delta=0.0001,
                                      patience=20, verbose=1, mode='auto')
        
        model.compile(loss=categorical_crossentropy,
                      optimizer=opt,
                      metrics=['acc'])
        
        return model, stop_callback, lr_callback

    ############################################################################
    ###########            PRINT INFORMATIONS            #######################
    ############################################################################
    def summary(self, main_attributes = 1, training_params = 1, other = 1):
        if main_attributes:
            print('Main model attributes:')
            print('\tpatch_size = ', self.patch_size)
            print('\tinput_size = ', self.input_size)
            print('\tpad_size = ', self.pad_size)
            print('\tpad_rotate_size = ', self.pad_rotate_size)
            print('\tfinal_layer_units = ', self.final_layer_units)
            print('\tpool_size = ', self.pool_size)

        # Training parameters
        if training_params:
            print('\nTraining parameters:')
            print('\tlearning_rate = ', self.learning_rate)
            print('\tepochs = ', self.epochs)
            print('\tbatch_size = ', self.batch_size)
            print('\tsteps_per_epoch = ', self.steps_per_epoch)

        # Other stuff
        if other:
            print('\nOther attributes:')
            print('\tNameWeights = ', self.NameWeights)
            print('\tSubmissionName = ', self.SubmissionName)
            print('\tPredictionName = ', self.PredictionName)

        # Model
        model, _, _ = self.CreateModel()
        print(f'\n{self.WhoAmI}')
        print('Keras model summary')
        model.summary() 