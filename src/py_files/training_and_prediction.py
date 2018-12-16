################################################################################
##################### CHOOSE THE MODEL #########################################
################################################################################
from model_1512 import *
from utilities import *

# NOTE: The imported file should define all the necessary parameters along with 
#       2 functions: CreateModel() and MinibatchGenerator(X,Y).

################################################################################
##################### TRAINING #################################################
################################################################################
def train(X, Y, validation_ratio = -1):    
    '''
    Generate an instance of the model an train the model on X, Y.
    If validation_ratio is a number between 0 and 1, X and Y are divided in train
    and validation sets.
    The function return the model. The training can be interrupted and in any 
    case the weights will be saved in a file.

    NOTE: this function must be called after having imported one of the model.py
          file
    '''
    # Create the model tha will be trained
    model, stop_callback, lr_callback = CreateModel()
    
    # Reproducibility + remember the deadline is the 20.12.2018
    np.random.seed(20122018) 
    
    # Training...
    if (validation_ratio <= 0) or (validation_ratio >= 1):
        #Some informations
        print('Training the model.')
        print('Training images shape: ', X.shape) 
        print(f'Epochs: {epochs}\nBatch_size: {batch_size} \nSteps per epoch: {steps_per_epoch}')
        print('No validation data.')
        try:
            model.fit_generator(MinibatchGenerator(X,Y),
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                verbose=1,
                                callbacks=[lr_callback, stop_callback])
        except KeyboardInterrupt:
            print('\n\nKeyboard interruption!\n\n')
            pass

    else:
        #Some informations
        print('Training the model.')
        print('Validation ratio: ', validation_ratio)
        X_tr, X_val, Y_tr, Y_val = split_data(X, Y, validation_ratio, seed = 1)
        print('Training images shape: ', X_tr.shape) 
        print('Validation images shape: ', X_val.shape) 
        print(f'Epochs: {epochs}\nBatch_size: {batch_size} \nSteps per epoch: {steps_per_epoch}')
        try:
            model.fit_generator(MinibatchGenerator(X_tr,Y_tr),
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                verbose=1,
                                callbacks=[lr_callback, stop_callback],
                                validation_data = MinibatchGenerator(X_val,Y_val),
                                validation_steps = 5,)
        except KeyboardInterrupt:
            print('\n\nKeyboard interruption!\n\n')
            pass
    

    model.save_weights(NameWeights)
    
    print(f'Training done, weights saved in: {NameWeights}')
    
    return model



def ContinueTrain(X, Y, NameOld, NameNew, epochs_cont, validation_ratio = -1, seed = 111):
    ''' This function allows to go on with the training of a model.
        The weights of the old model must be loaded and are saved in a file called
        NameOld.
        The new weights will be saved in a file called NameNew.
        The train will continue up to keyboard interruption or for epochs_cont
        epochs.
    '''
    # Create the model that will be trained
    model, stop_callback, lr_callback = CreateModel()
    
    # Load the old weights
    print('Loading weights of the model from: ', NameOld)
    model.load_weights(NameOld)

    # We need to change seed, so we pick different data
    np.random.seed(seed) 
    
    # Training...
    if (validation_ratio <= 0) or (validation_ratio >= 1):
        #Some informations
        print('Restarting training...')
        print('Training images shape: ', X.shape) 
        print(f'Epochs: {epochs_cont}\nBatch_size: {batch_size} \nSteps per epoch: {steps_per_epoch}')
        print('No validation data.')
        try:
            model.fit_generator(MinibatchGenerator(X,Y),
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs_cont,
                                verbose=1,
                                callbacks=[lr_callback, stop_callback])
        except KeyboardInterrupt:
            print('\n\nKeyboard interruption!\n\n')
            pass

    else:
        #Some informations
        print('Restarting training...')
        print('Validation ratio: ', validation_ratio)
        X_tr, X_val, Y_tr, Y_val = split_data(X, Y, validation_ratio, seed = 1)
        print('Training images shape: ', X_tr.shape) 
        print('Validation images shape: ', X_val.shape) 
        print(f'Epochs: {epochs_cont}\nBatch_size: {batch_size} \nSteps per epoch: {steps_per_epoch}')
        try:
            model.fit_generator(MinibatchGenerator(X_tr,Y_tr),
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs_cont,
                                verbose=1,
                                callbacks=[lr_callback, stop_callback],
                                validation_data = MinibatchGenerator(X_val,Y_val),
                                validation_steps = 5,)
        except KeyboardInterrupt:
            print('\n\nKeyboard interruption!\n\n')
            pass
    
    model.save_weights(NameNew)
    
    print(f'Training done, weights saved in: {NameNew}')
    
    return model


################################################################################
################# HANDLING THE RESULTS  ########################################
################################################################################


def ComputeLocalF1Score(X, Y, model):
    '''This function takes as input X and Y, arrays containing the images and 
    the gt_images NOT padded, and computes the F1 score of the prediction using
    the model.
    
    NOTE: this function is used especially when we use a model with 2 final units,
    where we weren't able to monitor the f1 score during the training. Hence we split 
    at the very beginning our data in train/validation and we use this function
    for the validation set, instead of relying on the function "train", with a
    validation_ratio. '''
    X = padding_imgs(np.array(X),pad_size)
    Y = np.asarray(Y) # To be sure we convert again
    N_valid = Y.shape[0]
    print('Number of validation images: ', N_valid)

    #Create suitable input/labels for the model
    val_inputs = imgs_to_windows(X,train_shape,patch_size,input_size)
    val_gt_patches = [img_crop(Y[i], patch_size, patch_size) 
                        for i in range(N_valid)]
    val_gt_patches =  np.asarray([val_gt_patches[i][j] 
                        for i in range(len(val_gt_patches)) 
                        for j in range(len(val_gt_patches[i]))])
    val_true_labels = np.asarray([value_to_class(np.mean(val_gt_patches[i])) 
                        for i in range(len(val_gt_patches))])

    #Predict
    print('Predicting... ')
    val_prediction = model.predict(val_inputs)
    print('Done!')

    #Compute f1_Score
    if final_layer_units == 1:
        val_predicted_labels = ( (val_prediction > 0.5) * 1 ).flatten()
    if final_layer_units == 2:
        val_predicted_labels = ((val_prediction[:,0] < val_prediction[:,1]) 
                                * 1 ).flatten()
    print('F1 score on validation set is:', 
                f1_score_sklearn(val_true_labels,val_predicted_labels))
    
    return


def PredictAndSubmit(NameWeights, SubmissionName, PredictionName,
                     root = '../Data'):
    '''This function loads the test images and performs the prediction.
       After the prediction, a submission file is generated and the prediction 
       is saved using pickle.
    '''
    print('Loading test images')
    test_images = np.asarray(pick_test_images(root))
    test_images = padding_imgs(np.array(test_images),pad_size)
    
    # Prepare the input for the prediction
    test_inputs = imgs_to_windows(test_images,608,patch_size,input_size)
    print('Inputs for the test are ready. Shape: ', test_inputs.shape)
    
    print('Load the weights of the model from: ', NameWeights)
    model, _,_ = CreateModel()
    model.load_weights(NameWeights)

    print('Predicting...')
    test_prediction = model.predict(test_inputs)
    print('Done!')
    
    print('Generating submission and prediction (pickle) file...')
    if final_layer_units == 1:
        test_predicted_labels = ( (test_prediction > 0.5) * 1 ).flatten()
    if final_layer_units == 2:
        test_predicted_labels = ( (test_prediction[:,0] < test_prediction[:,1])
                                 * 1 ).flatten()

    test_labels = test_predicted_labels.reshape(50,-1)
    MY_masks_to_submission(SubmissionName, test_labels)

    with open(PredictionName, 'wb') as f:
        pickle.dump(test_predicted_labels, f)

    print('Submission saved in: ', SubmissionName)
    print('Prediction saved in ', PredictionName)


def PredictAndPlot(img, model, PLOT = True):
    ''' Use the model to predict on the img. Return the predicted image. 
    If PLOT is True also a plot is produced.
    '''
    #Predict
    dim = img.shape[0]
    img_reshaped = img.reshape((1,dim,dim,3))
    img_reshaped = padding_imgs(img_reshaped,pad_size)
    inputs = imgs_to_windows(img_reshaped,dim,patch_size,input_size)
    prediction = model.predict(inputs)
    if final_layer_units == 1:
        predicted_labels = ( (prediction > 0.5) * 1 ).flatten()
    if final_layer_units == 2:
        predicted_labels = ((prediction[:,0] < prediction[:,1]) 
                                * 1 ).flatten()
    
    # Plot
    im = label_to_img(dim, dim, patch_size, patch_size, predicted_labels)
    if PLOT:
        plt.imshow( concatenate_images(img,im) )

    return im