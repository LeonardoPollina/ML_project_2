# Machine learning project 2 (CS-433)
### Option 2: Road Segmentation
This folder contains all the functions needed to develop our project on road segmentation of satellite aerial images. The aim of the project is to segment an RGB image into small patches, and to assign to each one of them a binary label: value 1 means that there is a street, while 0 represents everything else (grass, houses, parking lots, etc...).

### Final score and submission
Our best CrowdAI score was of <b>0.86</b>. The submission is made by <i>RolexMachineLearningCenter</i> and its ID is: 24383.

### Required setup/libraries
The project has been done using `Python 3.6.5`. Moreover, we rely on various libraries listed in the table below. 

| Library       | Version       |
| ------------- |---------------|
| NumPy       | 1.14.3        |
| Scipy       | 1.1.0         |
| Scikit-learn| 0.19.1        |
| Keras       | 2.2.4         |
| Tensorflow  | 1.12.0        |

Note: `Keras` could be used also with `Theano` as a backend; however, our code is based on `Tensorflow`. Make sure that you are using the correct backend.


# Generate the prediction
1. Check the version of your libraries and install the missing ones.
2. Open the `config.py` file and set the following parameters: <code>ROOT_DIR</code> (directory containing the set images), <code>NameWeights</code> (name of the file were the weights are saved) and <code>SubmissionName</code> (name of the file where the submission will be saved).
3. Note that the file containing the weights should be placed in the same folder as the script `run.py`.
4. Execute the script `run.py`. 
```python
python3 run.py
```

# Content description
We are going to illustrate the structure of the present folder, its sub-folders and the main files.

    .
    ├── Notebooks                       
    |   ├── Best_model_training.ipynb          # Training of our best model
    │   └── Logistic_regression.ipynb          # Logistic regression
    └── src
    │   ├── models                       
    │   |   ├── best_model.py                  # Our best model
    │   |   ├── model_RNN.py                   # Recurrent NN
    │   |   └── model_vgg_1_neuron.py          # VGG-like network with one neuron at the end  
    |   ├── f1_score.py                        # Implementation of F1-score
    |   ├── logistic_utilities.py              # Utilities for the logistic regression
    |   ├── training_and_prediction.py         # Functions to train and predict 
    |   └── utilities.py                       # General utilities
    ├── config.py                              # A couple of configuration parameters
    ├── ReadME.md                              # ReadMe
    ├── run.py                                 # Predict and generate a submission file
    └── WeigthsFinalModel                      # Weights of the final model

## Notebooks (folder)
    .                        
    ├── Notebooks                       
    │   └── ...        
    .
In this folder, there are the notebooks showing our main training procedures. These notebooks have already been run to avoid a waste of time for the reader. 

<ul>
<li><b>Best_model_training.ipynb</b></li>

This notebook shows the training procedure we applied (80/20 train/validation) to our best model, this model can be found in `src/models/best_model.py`. 

<li><b>Logistic_regression.ipynb</b></li>

This notebook shows the training procedure to find the best hyperparameters (degree of polynomial expansion and regularization parameter) of the regularized logistic regression. This classifier was used only at the beginning to have a starting F1-score to compare with the more advanced (and computational demanding) CNN models. 

</ul>

## src (folder)
    .                        
    ├── src                       
    │   └── ...        
    .
<ul>
<li><b>utilities.py</b>
    These functions are general utilities that do not rely on the class <code>MODEL_CLASS()</code>; the file is divided in the following sub-sections.
    <ul>
        <li><b>Given functions:</b></li> This section contains the functions given with the project or throughout the course laboratories.
        <li><b>Loading data:</b></li> Functions to handle the loading of the images.
        <li><b>Images manipulation:</b></li> Operations applied to the images (like padding or data augmentation).
        <li><b>Submission:</b></li> Functions used to generate the submission file from a prediction.
        <li><b>Post-processing:</b></li> Functions used to improve the F1-score given by the prediction. Some intuitive "continuity" criteria regarding the streets were applied. 
    </ul>
    </li>
<li><b>training_and_prediction.py</b></li>
    These functions rely on the class <code>MODEL_CLASS()</code>. These methods are used to train a Keras model, to generate a prediction and to handle the prediction results.
<li><b>f1_score.py</b></li>
    In Keras 2.0, F1-score has been removed from the available <code>metrics</code> of the sequential model's <code>compile</code> function. In this file our customized F1-score metric was implemented, based on the F1-score functions  of the old version of Keras. 
    <br><b>NOTE</b>: This can only be used in the case of a model with one neuron in the output layer.
<li><b>logistic_utilities.py</b></li>
    These are the functions used in order to train the logistic regression model.
</ul>

### models (folder)
    .                        
    ├── src                       
    │    ├── models
    |    └── ...
    .
Here you can find some of the models we tried. These models are defined using a single class called <code>MODEL_CLASS()</code>. According to the definition of its attributes, different models can be implemented.
<ul>
    <li><b>best_model.py:</b></li> 
    This is our final model. It consists on a VGG-like network with 2 classes in the final layer. The derivation of its architecture and its parameters is described in the report.
    <li><b>model_vgg_1_neuron.py:</b></li> 
    This is an example of a model that has only one neuron in the output layer. The two main differences with a model with 2 neurons in the last layer are the following: the returned label of the function <code>MinibatchGenerator(X,Y)</code> is binary and the F1-score is computed among the <code>metrics</code>. This allows a simpler validation procedure, since we can rely on the parameter <code>validation_data</code> when we call the sequential model's <code>fit_generator</code> function.
    <li><b>model_RNN.py:</b></li> 
    This is an example of one of the models containing alternative paths inside the architecture (i.e. Recurrent Neural Network).
</ul>
 
