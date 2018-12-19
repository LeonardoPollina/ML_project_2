from config import *
import sys
sys.path.insert(0,'./src/')
sys.path.insert(0,'./src/models')
from training_and_prediction import *
from utilities import *
print('-----------------------------------------------------------------------')
print('CONFIG')
print('Root directory: ', ROOT_DIR)
print('Weights filename: ', NameWeights)
print('Submission filename: ', SubmissionName)

print('\n\n')
print('-----------------------------------------------------------------------')
print('GENERATE THE MODEL')
from model_vgg_2_neurons_class import MODEL_CLASS
MODEL = MODEL_CLASS()
MODEL.summary(main_attributes=1, training_params=0, other=0)

print('\n\n')
print('-----------------------------------------------------------------------')
print('PREDICTION')
TempPredictionName = 'temp.pkl'
print('Weights are loaded from: ', NameWeights)
print('Predicting (this may take a few minutes)...')
PredictAndSubmit(MODEL, NameWeights, SubmissionName, TempPredictionName,
                     root_dir = ROOT_DIR, verbose = 0)
print('Done!')

print('\n\n')
print('-----------------------------------------------------------------------')
print('POST-PROCESS')
post_process_and_submit(TempPredictionName, SubmissionName, verbose = 0)
# Clean the temp prediction
os.remove(TempPredictionName)
print('Submission saved in: ', SubmissionName)