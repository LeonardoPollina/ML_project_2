print('Machine Learning (CS-433) project 2.')
print('Satellite aerial image segmentation.\n')
print('Group members: Nicola Ischia, Marion Perier, Leonardo Pollina.\n\n')

print('------------------------- IMPORT --------------------------------------')
print('Importing libraries...')
from config import *
import sys
sys.path.insert(0,'./src/')
sys.path.insert(0,'./src/models')
from training_and_prediction import *
from utilities import *

print('\n\n')
print('------------------------- CONFIG --------------------------------------')
print('Root directory: ', ROOT_DIR)
print('Weights filename: ', NameWeights)
print('Submission filename: ', SubmissionName)
print('\n\n')
print('-------------------- GENERATE THE MODEL -------------------------------')
from best_model import MODEL_CLASS
MODEL = MODEL_CLASS()
MODEL.summary(main_attributes=1, training_params=0, other=0)

print('\n\n')
print('----------------------- PREDICTION ------------------------------------')
# The function PredictAndSubmit will generate a pickle file that we don't want
# to keep in this case. We will simply save it in a temporary file and then
# delete this file.
TempPredictionName = 'temp_pred_file.pkl'
print('Weights are loaded from: ', NameWeights)
print('Predicting (this may take a few minutes)...')
PredictAndSubmit(MODEL, NameWeights, SubmissionName, TempPredictionName,
                     root_dir = ROOT_DIR, verbose = 0)
# Clean the temporary prediction file prediction
os.remove(TempPredictionName)
print('Done!')

print('\n\n')
print('-------------------- POST-PROCESSING ----------------------------------')
post_process_and_submit(TempPredictionName, SubmissionName, verbose = 0)
print('Submission saved in: ', SubmissionName)