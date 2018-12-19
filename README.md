# Machine learning project 2 (CS-433)
### Option 2: Road Segmentation
In this folder, there are all the functions needed to develop our project on road segmentation of satellite aerial images. The aim of the project is to segment an RGB image into small patches, and to assign to each one of them a binary label. Value 1 means that there is a street, while 0 represents everything else (trees, houses, parking lots, ecc...)

### Required setup/libraries
The project has been done using `Python 3.6.5`. Moreover, we rely on various libraries; in the table we list the ones that we used.

| Library       | Version       |
| ------------- |---------------|
| NumPy       | 1.14.3        |
| Scipy       | 1.1.0         |
| Scikit-learn| 0.19.1        |
| Keras       | 2.2.4         |
| Tensorflow  | 1.12.0        |

Note: `Keras` could be used also with `Theano` as a backend; however, our code is based on `Tensorflow`.


# Generate the prediction
1. Check the version of your libraries and install the missing ones.
2. Open the `config.py` file and select the following parameters: `ROOT_DIR` (directory containing the set images) and `SubmissionName` (name of the file where the submission will be saved).
3. Execute the script `run.py`. 

# Content description
We are going to illustrate the structure of the folder and then to describe the sub-folders and the main files contained in it.

## Folder structure

    .
    ├── run.py                          # Predict and generate a submission file
    ├── Notebooks                       
    │   ├── qualcosa1                   # commento1
    │   ├── qualcosa2                   # commento2
    │   └── Logistic_regression.ipynb   # Logistic regression
    └── src
    │    ├── models                       
    │    |   ├── nome_modello_finale      # Final model
    │    |   └── ...                      # Other models
    |    ├── utilities.py                 
    |    ├── logistic_utilities.py
    |    └── training_and_prediction.py
    └── ReadME.md

## Notebooks folder
In this folder, there are notebooks that show our main training procedures. They could be runned, but this is not suggested because of the long computational time that may be required.

<ul>
<li><b>Logistic_regression.ipynb</b></li>

Here it is shown the training procedure in order to find the best hyperparameters (degree of polynomial expansion and regularization parameter) of the regularized logistic regression. We used this classifier only at the very beginning, to have a starting F1-score to compare with the more advanced (and computational demanding) CNNs. 

<li><b>nome(final model)</b></li>

Final model

<li><b>nome(example of CNN with x final_layer_units)</b></li>

bla bla bla

</ul>

## src folder
<ul>
<li><b>utilities.py</b>
    These functions are general utilities that do not rely on the class `MODEL_CLASS()`; the file is divided in sub-sections.
    <ul>
        <li><b>Given functions:</b></li> This section contains the functions that we used but that have not been implemented by us. This functions have been given along with the project description, or throughout the course laboratories.
        <li><b>Post-processing:</b></li> Functions used to manipulate the prediction in order to improve the F1-score. We tried to apply some "continuity" criteria to improve our prediction.
    </ul>
    </li>
<li><b>training_and_prediction.py</b></li>
    These functions rely on the class `MODEL_CLASS()`. As the name of the file suggests, these methods are used to train a `Keras` model, generate a prediction and handle the prediction results.
<li><b>f1_Score.py</b></li>
    In `Keras 2.0`, F1-score has been removed from the available `metrics` of the sequential model's `compile` function. We implemented our customed F1-score metrics starting from the functions of the old version of `Keras`. This can be used in the case of a model with only one unit in the final layer.
<li><b>logistic_utilities.py</b></li>
    These are the functions used in order to train the logistic model.
</ul>

## models folder
Here you can find some of the models that we tried. The models are defined using a class called `MODEL_CLASS()`, and according to the attributes and the methods of this class different models can be defined.
<ul>
    <li><b>nome_modello_finale.py:</b></li> This is our final model
    <li><b>One_unit_model.py:</b></li> This is a model that has only one unit in the final layer. 
    <li><b>Recurrent_model.py:</b></li> An example of one of the models that we tried involving a recurrent neural network.
</ul>
 
