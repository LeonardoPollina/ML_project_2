from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import f1_score as f1_score_sklearn
from utilities import *

def extract_features(img):
    '''Extract 6-dimensional features consisting of average RGB color as well 
    as variance'''
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat


def extract_img_features(filename, patch_size = 16):
    '''Extract features for a given image'''
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features(img_patches[i]) for i in 
        range(len(img_patches))])
    return X


def build_k_indices(data, k_fold, seed):
    """
    Build k indices for k-fold.
    """
    num_row = data.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, lambda_):
    '''
    Cross validation using logistic regression computed on train and validation 
    sets defined by k_indices.
    Return f1 of the validation sets averaged on the k folds.
    '''
    folds = k_indices.shape[0]
    f1 = np.zeros(folds)
    for k in range(folds):
        #split the data in a train and a test set
        idx = k_indices[k]
        yval = y[idx]
        if len(x.shape) == 1:
            xval = x[idx]
        else:
            xval = x[idx,:]   
        ytr = np.delete(y,idx,0)
        xtr = np.delete(x,idx,0)
        #Log. regression
        logreg = linear_model.LogisticRegression(C=lambda_, 
                                                class_weight="balanced")
        logreg.fit(xtr, ytr)
        z = logreg.predict(xtr)
        #compute f1
        f1[k] = f1_score_sklearn(ytr,z)  
    return np.mean(f1)

def grid_search_hyperparam(y, tx, lambdas, degrees, verbose = 2):
    '''Grid search with cross validation used to estimate the best 
    hyperparameters for logistic regression, that are lambda and the degree for 
    polynomial expansion. 
    '''
    f1 = np.zeros((len(lambdas), len(degrees)))
    #We iterate on the hyperparameters to find the best combination
    for idx_lambda, lambda_ in enumerate(lambdas):
        if verbose >= 1 : print(f'Grid search ====> {idx_lambda + 1}/{len(lambdas)} lambda starts...')
        for idx_degree, degree in enumerate(degrees):
            #Degree augmentation
            poly = PolynomialFeatures(degree)
            x_augmented = poly.fit_transform(tx)
            k_indices = build_k_indices(y, 4, 1)
            #Cross validation with ridge regression
            f1_temp = cross_validation(y, x_augmented, k_indices, lambda_)
            #Corresponding accuracy saved
            f1[idx_lambda, idx_degree] = f1_temp
            if verbose == 2: print('Grid search ====> Lambda = %.2e, degree = %d, F1-score = %.3f' % (1/lambda_, degree, f1_temp))
    #Determine the best combination of hyperparameters
    max_f1 = np.max(f1)
    best_lambda = lambdas[ np.where( f1 == max_f1 )[0] ][0]
    best_degree = degrees[ np.where( f1 == max_f1 )[1] ][0]
    if verbose >= 1:
        print('\nGrid search terminated.')
        print('Best lambda: ',best_lambda)
        print('Best degree: ',best_degree)
        print('Max F1-score: ', max_f1)
    return best_lambda, best_degree, max_f1


def predict_and_submit_logistic(degree, log_model, SubmissionName, 
                        patch_size = 16, root_dir = '../../Data/', verbose = 1):
    '''Generate the submission file in case of logistic regression. 
    The test images are loaded and then the data to feed the log_model are 
    generated.

    The submission is saved in a file called SubmissionName.
    '''
    if verbose: print('Loading test images...')
    test_images = pick_test_images(root_dir)
    n_test = len(test_images)
    
    if verbose: print('Generating inputs from test images...')
    test_patches = [img_crop(test_images[i], patch_size, patch_size) 
                    for i in range(n_test)]
    test_patches = np.asarray([test_patches[i][j] 
                    for i in range(len(test_patches)) 
                    for j in range(len(test_patches[i]))])
    test_Features = np.asarray([ extract_features(test_patches[i]) 
                    for i in range(len(test_patches))])

    poly = PolynomialFeatures(degree)
    test_Features = poly.fit_transform(test_Features)

    if verbose: print('Predicting...')
    Z_test = log_model.predict(test_Features)

    Z_array_of_1Dmasks = Z_test.reshape(n_test,-1)
    MY_masks_to_submission(SubmissionName, Z_array_of_1Dmasks)

    if verbose: print('Submission saved in: ', SubmissionName)
    return