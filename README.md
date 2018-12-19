# Machine learning project 2 (CS-433)
### Option 2: Road Segmentation
In this folder, there are all the functions needed to develop our project on road segmentation of satellite aerial images. The aim of the project is to segment an RGB image into small patches, and to assign to each one of them a binary label. Value 1 means that there is a street, while 0 represents everything else (trees, houses, parking lots, etc...).

### Required setup/libraries
The project has been done using `Python 3.6.5`. Moreover, we rely on various libraries; in the table we list the ones that we used.

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
2. Open the `config.py` file and select the following parameters: <code>ROOT_DIR</code> (directory containing the set images) and <code>SubmissionName</code> (name of the file where the submission will be saved).
3. Execute the script `run.py`. 

# Content description
We are going to illustrate the structure of the folder and then to describe the sub-folders and the main files contained in it.

## Folder structure

    .
    ├── run.py                                    # Predict and generate a submission file
    ├── config.py                                 # A couple of configuration parameters
    ├── Notebooks                       
    │      ├── qualcosa1                          # Training of the final model
    │      ├── qualcosa2                          # Training with only one neuron at the end
    │      └── Logistic_regression.ipynb          # Logistic regression
    └── src
    │    ├── models                       
    │    |      ├── nome_modello_finale           # Final model
    │    |      └── ...                           # Other models
    |    ├── utilities.py                 
    |    ├── training_and_prediction.py
    |    ├── f1_score.py
    |    └── logistic_utilities.py
    └── ReadME.md

## Notebooks folder
In this folder, there are notebooks that show our main training procedures. They could be runned, but this is not suggested because of the long computational time that may be required.

<ul>
<li><b>Logistic_regression.ipynb</b></li>

Here it is shown the training procedure in order to find the best hyperparameters (degree of polynomial expansion and regularization parameter) of the regularized logistic regression. We used this classifier only at the very beginning, to have a starting F1-score to compare with the more advanced (and computational demanding) NN models. 

<li><b>nome(final model)</b></li>

Final model

<li><b>nome(example of CNN with x final_layer_units)</b></li>

bla bla bla

</ul>

## src folder

<ul>
<li><b>utilities.py</b>
    These functions are general utilities that do not rely on the class <code>MODEL_CLASS()</code>; the file is divided in sub-sections.
    <ul>
        <li><b>Given functions:</b></li> This section contains the functions that we used but that have not been implemented by us. This functions have been given along with the project description, or throughout the course laboratories.
        <li><b>Add-section(s?):</b></li> bla bla bla bla.
        <li><b>Post-processing:</b></li> Functions used to manipulate the prediction in order to improve the F1-score. We tried to apply some "continuity" criteria to improve our prediction.
    </ul>
    </li>
<li><b>training_and_prediction.py</b></li>
    These functions rely on the class <code>MODEL_CLASS()</code>. As the name of the file suggests, these methods are used to train a Keras model, generate a prediction and handle the prediction results.
<li><b>f1_score.py</b></li>
    In Keras 2.0, F1-score has been removed from the available <code>metrics</code> of the sequential model's <code>compile</code> function. We implemented in this file our customed F1-score metrics starting from the functions of the old version of Keras. 
    <b>NOTE</b>: This can be only used in the case of a model with one unit in the final layer.
<li><b>logistic_utilities.py</b></li>
    These are the functions used in order to train the logistic model.
</ul>

## models folder
Here you can find some of the models that we tried. The models are defined using a class called <code>MODEL_CLASS()</code>, and according to the attributes and the methods of this class different models can be defined.
<ul>
    <li><b>nome_modello_finale.py:</b></li> 
    This is our final model
    <li><b>One_unit_model.py:</b></li> 
    This is an example of a model that has only one unit in the final layer. At this level, the main differences are 2: the returned label of the function <code>MinibatchGenerator(X,Y)</code> are categorical with 2 classes, and the <code>metrics</code> include our function to compute F1-score.
    <li><b>Recurrent_model.py:</b></li> 
    An example of one of the models that we tried involving skip connections (i.e. recurrent neural network).
</ul>
 
