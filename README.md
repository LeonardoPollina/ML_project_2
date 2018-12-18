# Machine learning project 2 (CS-433)
### Option 2: Road Segmentation


# Required setup/libraries
The project has been done using `Python 3.6.5`. Morevoer, we rely on various libraries. Here we list the ones that we used:
* `NumPy 1.14.3`
* `Scipy 1.1.0`
* `Scikit-learn 0.19.1'
* `Keras 2.2.4`
* `Tensorflow 1.12.0`

Note: `Keras` could be used also with `Theano` as a backend, however, our code is based on `Tensorflow`.

For this choice of project task, we provide a set of satellite images acquired 
from GoogleMaps. We also provide ground-truth images where each pixel is labeled 
as road or background. 

Your task is to train a classifier to segment roads in these images, i.e. 
assigns a label `road=1, background=0` to each pixel.

Submission system environment setup:

1. The dataset is available from the 
[CrowdAI page](https://www.crowdai.org/challenges/epfl-ml-road-segmentation).

2. Obtain the python notebook `segment_aerial_images.ipynb` from this github 
folder, to see example code on how to extract the images as well as 
corresponding labels of each pixel.

The notebook shows how to use `scikit learn` to generate features from each 
pixel, and finally train a linear classifier to predict whether each pixel is 
road or background. Or you can use your own code as well. Our example code here 
also provides helper functions to visualize the images, labels and predictions. 
In particular, the two functions `mask_to_submission.py` and 
`submission_to_mask.py` help you to convert from the submission format to a 
visualization, and vice versa.

3. As a more advanced approach, try `tf_aerial_images.py`, which demonstrates 
the use of a basic convolutional neural network in TensorFlow for the same 
prediction task.

Evaluation Metric:
 [F1 score](https://en.wikipedia.org/wiki/F1_score)
