##
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

def ADD_BIAS_ROW(X_IN,ADD_BIAS):
    if ADD_BIAS:
        X_OUT = np.concatenate([X_IN, np.ones(shape=(1, X_IN.shape[1]))], axis=0)
    else:
        X_OUT = X_IN
    return X_OUT
def ADD_BIAS_COLUMN(X_IN,ADD_BIAS):
    if ADD_BIAS:
        X_OUT = np.concatenate([X_IN, np.ones(shape=(X_IN.shape[0], 1))], axis=1)
    else:
        X_OUT = X_IN
    return X_OUT
def REMOVE_BIAS_ROW(X_IN,ADD_BIAS):
    if ADD_BIAS:
        X_OUT = X_IN[0:-1,:]
    else:
        X_OUT = X_IN
    return X_OUT
def REMOVE_BIAS_COLUMN(X_IN,ADD_BIAS):
    if ADD_BIAS:
        X_OUT = X_IN[:,0:-1]
    else:
        X_OUT = X_IN
    return X_OUT
def resolve_complex_right_eigenvalues(E, W):
    eval = copy.deepcopy(np.diag(E))
    comp_modes = []
    comp_modes_conj = []
    for i1 in range(E.shape[0]):
        if np.imag(E[i1, i1]) != 0:
            print(i1)
            # Find the complex conjugate
            for i2 in range(i1 + 1, E.shape[0]):
                if eval[i2] == eval[i1].conj():
                    break
            # i1 and i2 are the indices of the complex conjugate eigenvalues
            comp_modes.append(i1)
            comp_modes_conj.append(i2)
            E[i1, i1] = np.real(eval[i1])
            E[i2, i2] = np.real(eval[i1])
            E[i1, i2] = np.imag(eval[i1])
            E[i2, i1] = - np.imag(eval[i1])
            u1 = copy.deepcopy(np.real(W[:, i1:i1 + 1]))
            w1 = copy.deepcopy(np.imag(W[:, i1:i1 + 1]))
            W[:, i1:i1 + 1] = u1
            W[:, i2:i2 + 1] = w1
    E_out = np.real(E)
    W_out = np.real(W)
    return E_out, W_out, comp_modes, comp_modes_conj

# Importing all necessary information
SYSTEM_NO = 200
ocdeepDMD_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
# Import the training data
with open(ocdeepDMD_data_path,'rb') as handle:
    dict_DMD_train = pickle.load(handle) # already scaled, it is in transposed format though not stated
# Since the data is scaled, we use the affine transformation theorem to impose a bias constraint on the model
dict_DMD_train['Xp'] = ADD_BIAS_COLUMN(dict_DMD_train['Xp'],True)
dict_DMD_train['Xf'] = ADD_BIAS_COLUMN(dict_DMD_train['Xf'],True)
# Import the full dataset and the indices
with open(original_data_path,'rb') as handle:
    dict_data_original = pickle.load(handle)
# Import the ordered indices - Last two indices are used as testing data
with open(indices_path,'rb') as handle:
    ls_data_indices = pickle.load(handle)
# Setup the prediction dictionary for all datasets
dict_DMD_results = {}
for i in ls_data_indices:
    dict_DMD_results[i] = {}
    dict_DMD_results[i]['Xp'] = np.array(dict_data_original[i]['df_X_TPM'].iloc[:,0:-1])
    dict_DMD_results[i]['Xf'] = np.array(dict_data_original[i]['df_X_TPM'].iloc[:, 1:])
    dict_DMD_results[i]['Yp'] = np.array(dict_data_original[i]['Y'].iloc[:, 0:-1])
    dict_DMD_results[i]['Yf'] = np.array(dict_data_original[i]['Y'].iloc[:, 1:])
# The scalers can be called directly using functions

## --------------------------------------------------------------------------------
# DMD Train with Lasso Regression and k-fold cross validation

# Notes - This does not require a validation data. We use each one of the k-folds as the validation to draw a statistic
# on which is the most robust hyperparameter (\lambda - the lasso regularization parameter)

NO_OF_FOLDS = 14

kf = KFold(n_splits=NO_OF_FOLDS, shuffle=False, random_state=None)
my_scorer = make_scorer(r2_score,multioutput='uniform_average')
# print(cross_val_score(LinearRegression(fit_intercept=False), dict_DMD_train['Xp'], dict_DMD_train['Xf'], cv=kf.split(dict_DMD_train['Xp']),scoring=my_scorer))
# print(cross_val_score(Lasso(alpha= 0.02, fit_intercept=False, max_iter=50000), dict_DMD_train['Xp'], dict_DMD_train['Xf'],cv=kf.split(dict_DMD_train['Xp']), scoring=my_scorer))

dict_stats = {}
for alpha in np.arange(0.02,0.1,0.02):
    dict_stats[alpha] = {}
    if alpha ==0:
        a =cross_val_score(LinearRegression(fit_intercept=False), dict_DMD_train['Xp'], dict_DMD_train['Xf'], cv=kf.split(dict_DMD_train['Xp']),scoring=my_scorer)
    else:
        a = cross_val_score(Lasso(alpha= alpha, fit_intercept=False, max_iter=50000), dict_DMD_train['Xp'], dict_DMD_train['Xf'],cv=kf.split(dict_DMD_train['Xp']), scoring=my_scorer)
    for i in range(NO_OF_FOLDS):
        dict_stats[alpha][i] = a[i]
    print('[STATUS COMPLETE] alpha = ',alpha)

df_stats = pd.DataFrame(dict_stats)
print(df_stats)

## Saving results of lasso regression
# Scheme 1 : save the results to a new pickle
with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_LasssoRegression_Run2_Results.pickle','wb') as handle:
    pickle.dump(df_stats, handle)
# # Scheme 2 : append the results to an existing pickle file [only use if you have the same number of folds (kfold cross validation)]
# Opening the old lasso regression results
# with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_LasssoRegression_Results.pickle','rb') as handle:
#     df_stats1 = pickle.load(handle)
# Concatenating to the new ones
# df_stats_new = pd.concat([df_stats1,df_stats],axis=1).sort_index(axis=1)
# Saving the concatenated lasso regression results
# with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_LasssoRegression_Results.pickle','wb') as handle:
#     pickle.dump(df_stats_new, handle)

##
with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_LasssoRegression_Results.pickle','rb') as handle:
    df_stats1 = pickle.load(handle)
df_stats_filt = np.maximum(df_stats1, 0)

for i in df_stats_filt.columns:
    if i>0.5:
        df_stats_filt = df_stats_filt.drop(columns=[i])
for i in df_stats_filt.index:
    if np.sum(df_stats_filt.loc[i,:])==0:
        df_stats_filt = df_stats_filt.drop(index=[i])
plt.plot(df_stats_filt.columns, df_stats_filt.T, '.',color= '#ff7f0e')
# plt.plot(df_stats_filt.columns,df_stats_filt.median(axis=0),color = '#ff7f0e')
plt.plot(df_stats_filt.columns,df_stats_filt.mean(axis=0), color = '#1f77b4',marker='.',markersize = 10,alpha = 1)
plt.errorbar(df_stats_filt.columns,df_stats_filt.mean(axis=0),df_stats_filt.std(axis=0), color = '#1f77b4', capsize = 8,fmt='.',alpha = 0.8)
plt.xlabel('Lasso hyperparameter ($\lambda$)')
plt.ylabel('$r^2$')
plt.yticks([0,0.3,0.6,0.9])
plt.xticks([0,0.1,0.2,0.3,0.4])
plt.xlim([-0.005,0.4])
plt.ylim([-0.005,1])
plt.show()
## Fit lasso regression models and

model_lin_reg = LinearRegression(fit_intercept=False).fit()