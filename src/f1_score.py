################################################################################
# This code is taken from the GitHub repository of Keras, that has removed the #
# F1-score metric since the version 2.0. We have slightly modified the code to #
# be able to monitor the F1 score instead of the accuracy, since our goal is   #
# to have a good F1-score.                                                     #
#                                                                              #
# We tried to extend the changes to handle even in the case of a model with 2  #
# neurons in the final layer, but without success.                             #
################################################################################

from keras import backend as K

def precision(y_true, y_pred0):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    
    # OUR FIRST CHANGE TO THE ORIGINAL CODE FROM KERAS STARTS HERE... ##########
    y_pred = y_pred0 > 0.5 
    y_pred = K.cast(y_pred, 'float32')
    # ... AND ENDS HERE ########################################################
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred0):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    
    # OUR SECOND CHANGE TO THE ORIGINAL CODE FROM KERAS STARTS HERE... #########
    y_pred = y_pred0 > 0.5 
    y_pred = K.cast(y_pred, 'float32')
    # ... AND ENDS HERE ########################################################
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
     # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def f1_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, 1)