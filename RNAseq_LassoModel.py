import RNAseq_data_preprocessing_functions as rnaf
import ocdeepdmd_simulation_examples_helper_functions as oc
import pickle
import random
import numpy as np
import pandas as pd
import os
import shutil
import random
import matplotlib.pyplot as plt
import copy
import itertools
import seaborn as sb
import tensorflow as tf

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
from sklearn.metrics import make_scorer,r2_score
pd.set_option("display.max_rows", None, "display.max_columns", None)
plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22

# Preprocessing files
SYSTEM_NO = 401
ALL_CONDITIONS = ['MX']
# SYSTEM_NO = 400
# ALL_CONDITIONS = ['MX','MN']

original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'


# Indices [train and test]
with open(indices_path,'rb') as handle:
    ls_data_indices = pickle.load(handle)
ls_train_indices = ls_data_indices[0:14]
ls_test_indices = ls_data_indices[14:16]

# Datasets [sorted as scaled and unscaled] and Conditions
with open(original_data_path,'rb') as handle:
    dict_data_original = pickle.load(handle)

n_genes = len(dict_data_original[ALL_CONDITIONS[0]][ls_data_indices[0]]['df_X_TPM'])

dict_empty_all_conditions ={}
for COND in ALL_CONDITIONS:
    dict_empty_all_conditions[COND] = {}

dict_scaled_data = copy.deepcopy(dict_empty_all_conditions)
dict_unscaled_data = copy.deepcopy(dict_empty_all_conditions)
for COND,i in itertools.product(ALL_CONDITIONS,ls_data_indices):
    dict_intermediate = oc.scale_data_using_existing_scaler_folder(
        {'Xp': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 0:-1]).T,
         'Xf': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 1:]).T,
         'Yp': np.array(dict_data_original[COND][i]['Y'].iloc[:, 0:-1]).T,
         'Yf': np.array(dict_data_original[COND][i]['Y'].iloc[:, 1:]).T}, SYSTEM_NO)
    dict_scaled_data[COND][i] = {'XpT': dict_intermediate['Xp'], 'XfT': dict_intermediate['Xf'],
                                 'YpT': dict_intermediate['Yp'], 'YfT': dict_intermediate['Yf']}
    dict_unscaled_data[COND][i] = {'XpT': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 0:-1]).T,
                                   'XfT': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 1:]).T,
                                   'YpT': np.array(dict_data_original[COND][i]['Y'].iloc[:, 0:-1]).T,
                                   'YfT': np.array(dict_data_original[COND][i]['Y'].iloc[:, 1:]).T}

## --------------------------------------------------------------------------------
# DMD Train with Lasso Regression and k-fold cross validation

# Notes - This does not require a validation data. We use each one of the k-folds as the validation to draw a statistic
# on which is the most robust hyperparameter (\lambda - the lasso regularization parameter)

XpT_train=[]
XfT_train=[]
for COND,i in itertools.product(ALL_CONDITIONS,ls_train_indices[0:14]):
    try:
        XpT_train = np.concatenate([XpT_train, np.array(dict_data_original[COND][i]['df_X_TPM'])[:,0:-1].T],axis=0)
        XfT_train = np.concatenate([XfT_train, np.array(dict_data_original[COND][i]['df_X_TPM'])[:, 1:].T], axis=0)
    except:
        XpT_train = np.array(dict_data_original[COND][i]['df_X_TPM'])[:, 0:-1].T
        XfT_train = np.array(dict_data_original[COND][i]['df_X_TPM'])[:, 1:].T

XpT_test=[]
XfT_test=[]
for COND,i in itertools.product(ALL_CONDITIONS,ls_test_indices[0:14]):
    try:
        XpT_test = np.concatenate([XpT_test, np.array(dict_data_original[COND][i]['df_X_TPM'])[:,0:-1].T],axis=0)
        XfT_test = np.concatenate([XfT_test, np.array(dict_data_original[COND][i]['df_X_TPM'])[:, 1:].T], axis=0)
    except:
        XpT_test = np.array(dict_data_original[COND][i]['df_X_TPM'])[:, 0:-1].T
        XfT_test = np.array(dict_data_original[COND][i]['df_X_TPM'])[:, 1:].T

NO_OF_FOLDS = 7
kf = KFold(n_splits=NO_OF_FOLDS, shuffle=False, random_state=None)
my_scorer = make_scorer(r2_score,multioutput='uniform_average')
# print(cross_val_score(LinearRegression(fit_intercept=False), dict_DMD_train['Xp'], dict_DMD_train['Xf'], cv=kf.split(dict_DMD_train['Xp']),scoring=my_scorer))
# print(cross_val_score(Lasso(alpha= 0.02, fit_intercept=False, max_iter=50000), dict_DMD_train['Xp'], dict_DMD_train['Xf'],cv=kf.split(dict_DMD_train['Xp']), scoring=my_scorer))

dict_stats = {}
for alpha in np.arange(0.0,0.5,0.5):
    dict_stats[alpha] = {}
    if alpha ==0:
        a =cross_val_score(LinearRegression(fit_intercept=False), dict_DMD_train['Xp'], dict_DMD_train['Xf'], cv=kf.split(dict_DMD_train['Xp']),scoring=my_scorer)
    else:
        a = cross_val_score(Lasso(alpha= alpha, fit_intercept=False, max_iter=50000), dict_DMD_train['Xp'], dict_DMD_train['Xf'],cv=kf.split(dict_DMD_train['Xp']), scoring=my_scorer)
    for i in range(NO_OF_FOLDS):
        dict_stats[alpha][i] = a[i]
    print('[STATUS COMPLETE] alpha = ',alpha)
    print(a)

df_stats = pd.DataFrame(dict_stats)
print(df_stats)


## Saving results of lasso regression

try:
    # Scheme 2 : append the results to an existing pickle file [only use if you have the same number of folds (kfold cross validation)]
    # Opening the old lasso regression results
    with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_LasssoRegression_Results.pickle','rb') as handle:
        df_stats1 = pickle.load(handle)
    # Concatenating to the new ones
    df_stats_new = pd.concat([df_stats1,df_stats],axis=1).sort_index(axis=1)
    # Saving the concatenated lasso regression results
    with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_LasssoRegression_Results.pickle','wb') as handle:
        pickle.dump(df_stats_new, handle)
except:
    # Scheme 1 : save the results to a new pickle
    with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_LasssoRegression_Results.pickle', 'wb') as handle:
        pickle.dump(df_stats, handle)