# Machine learning project 2 (CS-433)
### Option 2: Road Segmentation
In this folder, there are all the functions needed to develop our project on road segmentation of satellite aerial images. The aim of the project is to segment an RGB image into small patches, and to assign to each one of them a binary label. Value 1 means that there is a street, while 0 represents everything else (trees, houses, parking lots, ecc...)

### Required setup/libraries
The project has been done using `Python 3.6.5`. Moreover, we rely on various libraries. In the table we list the ones that we used.

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

## `Notebooks` folder
    .                        
    ├── Notebooks                       
    │   └── ...                  
    .
 
    
## `utilities.py`
## `models` folder
